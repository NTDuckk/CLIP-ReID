import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
import collections
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
from loss.make_loss import shape_consistency_loss

def do_train_stage1(cfg,
             model,
             train_loader_stage1,
             optimizer,
             scheduler,
             local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD 

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    # Resolve underlying model (handle DataParallel)
    _model = model.module if hasattr(model, 'module') else model
    use_shape = _model.use_shape
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    image_features = []
    labels = []
    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1):
            img = img.to(device)
            target = vid.to(device)
            with amp.autocast(enabled=True):
                image_feature = model(img, target, get_image = True)
                for i, img_feat in zip(target, image_feature):
                    labels.append(i)
                    image_features.append(img_feat.cpu())
        labels_list = torch.stack(labels, dim=0).cuda() #N
        image_features_list = torch.stack(image_features, dim=0).cuda()

        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch

        # Pre-compute bipolar shape scores (frozen, from CLIP zero-shot)
        if use_shape:
            shape_text_feats = _model.shape_text_feats  # [32, 512]
            img_norm = F.normalize(image_features_list.float(), dim=-1)
            text_norm = F.normalize(shape_text_feats.float(), dim=-1)
            all_scores = img_norm @ text_norm.t()                  # [N, 32]
            bipolar_scores_list = all_scores[:, 0::2] - all_scores[:, 1::2]  # [N, 16]
            logger.info("Computed bipolar shape scores for {} images".format(num_image))
        else:
            bipolar_scores_list = None
    del labels, image_features

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        iter_list = torch.randperm(num_image).to(device)
        for i in range(i_ter+1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i*batch:(i+1)* batch]
            else:
                b_list = iter_list[i*batch:num_image]
            
            target = labels_list[b_list]
            image_features = image_features_list[b_list]

            # Get bipolar scores for this batch
            if use_shape and bipolar_scores_list is not None:
                bp_batch = bipolar_scores_list[b_list]
            else:
                bp_batch = None

            with amp.autocast(enabled=True):
                text_features = model(label=target, get_text=True, bipolar_scores=bp_batch)
            loss_i2t = xent(image_features, text_features, target, target)
            loss_t2i = xent(text_features, image_features, target, target)

            loss = loss_i2t + loss_t2i

            # Shape consistency loss
            if use_shape and bp_batch is not None:
                loss_sc = shape_consistency_loss(bp_batch, target)
                loss = loss + cfg.MODEL.SHAPE_CONSISTENCY_LOSS_WEIGHT * loss_sc

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), len(train_loader_stage1),
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))
