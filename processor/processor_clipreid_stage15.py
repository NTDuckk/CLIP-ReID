import logging
import os
import torch
import torch.nn.functional as F
from torch.cuda import amp

from utils.meter import AverageMeter
from loss.supcontrast import SupConLoss


def do_train_stage15(cfg,
                     model,
                     train_loader_stage15,
                     optimizer,
                     scheduler,
                     local_rank):
    """
    Stage15 (no shape label):
    - ONLY optimize shape_ctx (2 tokens/ID) via:
        (1) distill teacher(32 shape prompts) -> student(V V)
        (2) regularize alignment with Li2t + Lt2i (SupCon) between img and student_txt
    """

    device = torch.device(f"cuda:{local_rank}") if local_rank is not None else torch.device("cuda")
    epochs = cfg.SOLVER.STAGE15.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE15.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE15.CHECKPOINT_PERIOD
    batch_size = cfg.SOLVER.STAGE15.IMS_PER_BATCH

    # distill hyperparams
    T = getattr(cfg.SOLVER.STAGE15, "SHAPE_T", 2.0)
    tau = getattr(cfg.SOLVER.STAGE15, "SHAPE_TAU", 0.05)

    # weights
    lambda_distill = getattr(cfg.SOLVER.STAGE15, "LAMBDA_SHAPE_DISTILL", 1.0)
    lambda_align = getattr(cfg.SOLVER.STAGE15, "LAMBDA_ALIGN", 0.1)

    logger = logging.getLogger("transreid.train")
    logger.info("start training stage15 (shape distill + i2t/t2i regularizer)")

    # ============================================================
    # 0) Freeze all params except prompt_learner.shape_ctx
    # ============================================================
    model.to(device)

    # unwrap DP/DDP safely
    m = model.module if hasattr(model, "module") else model

    # IMPORTANT: ensure Stage15 student prompt uses "shape" mode ("... who is V V.")
    if hasattr(m, "prompt_learner") and hasattr(m.prompt_learner, "set_prompt_mode"):
        m.prompt_learner.set_prompt_mode("shape")

    for _, p in model.named_parameters():
        p.requires_grad_(False)

    if hasattr(m, "prompt_learner") and hasattr(m.prompt_learner, "shape_ctx"):
        m.prompt_learner.shape_ctx.requires_grad_(True)
    else:
        raise AttributeError("model.prompt_learner.shape_ctx not found. Please use the modified PromptLearner.")

    # eval mode to stabilize cache (ok because only shape_ctx has grad)
    model.eval()

    # ============================================================
    # 1) Cache image features (no_grad) like stage1
    # ============================================================
    logger.info("stage15: caching image features (no_grad)")
    image_features_list = []
    labels_list = []

    with torch.no_grad():
        for batch in train_loader_stage15:
            img, pid, camid, viewid = batch
            img = img.to(device, non_blocking=True)
            pid = pid.to(device, non_blocking=True)

            image_feat = model(img, get_image=True)  # (B, Dproj)
            image_features_list.append(image_feat.detach().cpu())
            labels_list.append(pid.detach().cpu())

    image_features_all = torch.cat(image_features_list, dim=0)  # CPU
    labels_all = torch.cat(labels_list, dim=0)                  # CPU
    num_samples = labels_all.shape[0]
    logger.info(f"stage15 cache done: {num_samples} samples")

    # losses/meters
    xent = SupConLoss(device)
    loss_meter = AverageMeter()
    loss_distill_meter = AverageMeter()
    loss_align_meter = AverageMeter()
    used_meter = AverageMeter()
    conf_meter = AverageMeter()

    scaler = amp.GradScaler(enabled=True)

    # ============================================================
    # 2) Train loop
    # ============================================================
    for epoch in range(1, epochs + 1):
        scheduler.step(epoch)

        loss_meter.reset()
        loss_distill_meter.reset()
        loss_align_meter.reset()
        used_meter.reset()
        conf_meter.reset()

        perm = torch.randperm(num_samples).split(batch_size)
        for n_iter, idx in enumerate(perm):
            idx_cpu = idx.cpu()
            img_feat = image_features_all[idx_cpu].to(device, non_blocking=True)  # [B,D]
            target = labels_all[idx_cpu].to(device, non_blocking=True)            # [B]

            optimizer.zero_grad(set_to_none=True)

            # --- get scale safely (works for DDP/DP too) ---
            m = model.module if hasattr(model, "module") else model
            if hasattr(m, "logit_scale") and m.logit_scale is not None:
                scale = m.logit_scale.exp().detach().float()
                scale = scale.clamp(1e-3, 100.0)
            else:
                scale = torch.tensor(1.0, device=device, dtype=torch.float32)

            with amp.autocast(enabled=True):
            # ------------------------------------------------
            # Teacher bank + teacher probs (no grad)
            # ------------------------------------------------
                with torch.no_grad():
                    teacher_txt = m.get_shape_teacher_text_features(target)   # [B,32,D]
                    teacher_n = F.normalize(teacher_txt.float(), dim=2)        # fp32

                    img_n = F.normalize(img_feat.float(), dim=1)              # fp32

                    teacher_logits_fp32 = (scale * torch.einsum("bd,bkd->bk", img_n, teacher_n)).float()
                    
                    # GIẢI PHÁP 3: Dùng temperature thấp hơn cho teacher để phân phối "sharp" hơn
                    T_teacher = T * 0.5  # Teacher temperature thấp hơn student
                    p_teacher = F.softmax(teacher_logits_fp32 / T_teacher, dim=1).detach()     # [B,32] fp32

                    conf = p_teacher.max(dim=1).values                                  # [B]
                    # GIẢI PHÁP 1: Thay hard mask bằng soft weighting
                    # weights = torch.sigmoid((conf - tau) * 20)  # Sigmoid với slope cao
                    weights = conf  # Đơn giản: dùng confidence làm weight trực tiếp

                # ------------------------------------------------
                # Student text (grad -> shape_ctx)
                # ------------------------------------------------
                # (student prompt is shape-mode: "... who is V V.")
                student_txt = m.get_shape_student_text_features(target)        # [B,D]
                student_n = F.normalize(student_txt.float(), dim=1)            # fp32

                student_logits_fp32 = (scale * torch.einsum("bd,bkd->bk", student_n, teacher_n)).float()
                log_p_student = F.log_softmax(student_logits_fp32 / T, dim=1)  # fp32

                # GIẢI PHÁP 5: KL divergence với label smoothing
                alpha = 0.1  # Label smoothing coefficient
                p_teacher_smooth = p_teacher * (1 - alpha) + alpha / p_teacher.size(1)
                kl = (p_teacher_smooth * (torch.log(p_teacher_smooth + 1e-12) - log_p_student)).sum(dim=1)  # [B]
                
                # GIẢI PHÁP 1 (tiếp): Dùng weighted loss thay vì masked loss
                # loss_distill = (kl * weights).sum() / (weights.sum() + 1e-6)
                # Hoặc đơn giản hơn: dùng tất cả samples với trọng số bằng nhau
                loss_distill = kl.mean()
                loss_distill = loss_distill * (T * T)

                # ------------------------------------------------
                # Alignment regularizer (Li2t + Lt2i)
                # ------------------------------------------------
                loss_i2t = xent(img_feat, student_txt, target, target)
                loss_t2i = xent(student_txt, img_feat, target, target)
                loss_align = loss_i2t + loss_t2i

                loss = lambda_distill * loss_distill + lambda_align * loss_align
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # meters
            loss_meter.update(loss.item(), target.size(0))
            loss_distill_meter.update(loss_distill.item(), target.size(0))
            loss_align_meter.update(loss_align.item(), target.size(0))
            conf_meter.update(conf.mean().item(), target.size(0))
            used_meter.update(mask.mean().item(), target.size(0))

            if (n_iter + 1) % log_period == 0:
                logger.info(
                    f"Stage15 Epoch[{epoch}] Iter[{n_iter+1}/{len(perm)}] "
                    f"Loss: {loss_meter.avg:.4f} "
                    f"(distill: {loss_distill_meter.avg:.4f}, align: {loss_align_meter.avg:.4f}) "
                    f"(conf: {conf_meter.avg:.3f}, used_ratio: {used_meter.avg:.3f}) "
                    f"Lr: {optimizer.param_groups[0]['lr']:.2e} "
                    f"T={T}, tau={tau}, w={lambda_distill}/{lambda_align}"
                )

        # save ckpt
        if (epoch % checkpoint_period == 0) or (epoch == epochs):
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            ckpt = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_stage15_shape_{epoch}.pth")
            torch.save(model.state_dict(), ckpt)
            logger.info(f"Stage15 checkpoint saved to {ckpt}")

    logger.info("stage15 training done.")
