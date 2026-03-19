import logging
import os
import time
from datetime import timedelta

import torch
import torch.nn as nn
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F

from utils.meter import AverageMeter
from loss.supcontrast import SupConLoss


def do_train_stage1(cfg,
             model,
             train_loader_stage1,
             optimizer,
             scheduler,
             local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD
    device = "cuda"

    logger = logging.getLogger("transreid.train")
    logger.info("start training stage1 (inversion prompt learning)")

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    core_model = model.module if isinstance(model, nn.DataParallel) else model

    loss_meter = AverageMeter()
    loss_i2t_meter = AverageMeter()
    loss_t2i_meter = AverageMeter()

    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        loss_i2t_meter.reset()
        loss_t2i_meter.reset()

        scheduler.step(epoch)
        model.train()

        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1):
            optimizer.zero_grad()

            img = img.to(device, non_blocking=True)
            target = vid.to(device, non_blocking=True)
            target_cam = target_cam.to(device, non_blocking=True)
            target_view = target_view.to(device, non_blocking=True)

            with amp.autocast(enabled=True):
                image_feature, text_features = model(
                    x=img,
                    label=target,
                    get_text_inversion=True,
                    cam_label=target_cam,
                    view_label=target_view
                )

                image_feature = F.normalize(image_feature.float(), dim=1)
                text_features = F.normalize(text_features.float(), dim=1)

                loss_i2t = xent(image_feature, text_features, target, target)
                loss_t2i = xent(text_features, image_feature, target, target)
                loss = loss_i2t + loss_t2i

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = img.size(0)
            loss_meter.update(loss.item(), batch_size)
            loss_i2t_meter.update(loss_i2t.item(), batch_size)
            loss_t2i_meter.update(loss_t2i.item(), batch_size)

            torch.cuda.synchronize()

            if (n_iter + 1) % log_period == 0:
                if hasattr(scheduler, "_get_lr"):
                    base_lr = scheduler._get_lr(epoch)[0]
                else:
                    base_lr = optimizer.param_groups[0]["lr"]

                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Li2t: {:.3f}, Lt2i: {:.3f}, Base Lr: {:.2e}".format(
                        epoch,
                        n_iter + 1,
                        len(train_loader_stage1),
                        loss_meter.avg,
                        loss_i2t_meter.avg,
                        loss_t2i_meter.avg,
                        base_lr
                    )
                )

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_stage1_{}.pth".format(epoch))
                    )
            else:
                torch.save(
                    model.state_dict(),
                    os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_stage1_{}.pth".format(epoch))
                )

    logger.info("Computing averaged text features per ID from stage1 (old prototype flow)...")
    model.eval()

    if isinstance(model, nn.DataParallel):
        num_classes = model.module.num_classes
    else:
        num_classes = model.num_classes

    text_feature_sum = None
    text_feature_count = torch.zeros(num_classes, device=device, dtype=torch.float32)

    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1):
            img = img.to(device, non_blocking=True)
            target = vid.to(device, non_blocking=True).long()

            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device, non_blocking=True)
            else:
                target_cam = None

            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device, non_blocking=True)
            else:
                target_view = None

            with amp.autocast(enabled=True):
                _, text_features = model(
                    x=img,
                    label=target,
                    get_text_inversion=True,
                    cam_label=target_cam,
                    view_label=target_view
                )

            text_features = text_features.float()

            if text_feature_sum is None:
                feat_dim = text_features.size(1)
                text_feature_sum = torch.zeros(
                    num_classes,
                    feat_dim,
                    device=device,
                    dtype=text_features.dtype
                )

            text_feature_sum.index_add_(0, target, text_features)
            text_feature_count.index_add_(
                0,
                target,
                torch.ones(target.size(0), device=device, dtype=text_feature_sum.dtype)
            )

    if text_feature_sum is None:
        raise RuntimeError("Failed to build stage1 prototypes: no text features were accumulated.")

    avg_text_features = torch.zeros_like(text_feature_sum)
    valid_mask = text_feature_count > 0
    avg_text_features[valid_mask] = (
        text_feature_sum[valid_mask] /
        text_feature_count[valid_mask].unsqueeze(1)
    )

    logger.info("Built stage1 prototypes for {} / {} classes".format(
        int(valid_mask.sum().item()), num_classes
    ))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))

    return avg_text_features.detach()