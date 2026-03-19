import torch

def make_optimizer_1stage(cfg, model):
    # Freeze everything first
    for _, value in model.named_parameters():
        value.requires_grad_(False)

    params = []
    keys = []

    for key, value in model.named_parameters():
        # Only optimize the 3 IM2TEXT branches inside inversion_prompt_learner
        if "inversion_prompt_learner" in key:
            value.requires_grad_(True)
            lr = cfg.SOLVER.STAGE1.BASE_LR
            weight_decay = cfg.SOLVER.STAGE1.WEIGHT_DECAY
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]

    if len(params) == 0:
        raise RuntimeError("No trainable parameters found for stage1 under 'inversion_prompt_learner'")

    print("Stage1 trainable params:")
    for key in keys:
        print("  ", key)

    if cfg.SOLVER.STAGE1.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE1.OPTIMIZER_NAME)(
            params, momentum=cfg.SOLVER.STAGE1.MOMENTUM
        )
    elif cfg.SOLVER.STAGE1.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.SOLVER.STAGE1.BASE_LR,
            weight_decay=cfg.SOLVER.STAGE1.WEIGHT_DECAY
        )
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE1.OPTIMIZER_NAME)(params)

    return optimizer


# def make_optimizer_1stage(cfg, model):
#     # freeze all first
#     for _, value in model.named_parameters():
#         value.requires_grad_(False)

#     params = []
#     keys = []

#     # 1) inversion MLPs: main trainable part
#     for key, value in model.named_parameters():
#         if "inversion_prompt_learner" in key:
#             value.requires_grad_(True)
#             params += [{
#                 "params": [value],
#                 "lr": cfg.SOLVER.STAGE1.BASE_LR,
#                 "weight_decay": cfg.SOLVER.STAGE1.WEIGHT_DECAY
#             }]
#             keys += [f"{key} | lr={cfg.SOLVER.STAGE1.BASE_LR}"]

#     # 2) lightly unfreeze a small part of visual encoder
#     visual_lr = cfg.SOLVER.STAGE1.BASE_LR * 0.1   # thử 0.1x trước
#     visual_open_keywords = [
#         "image_encoder.ln_post",
#         "image_encoder.proj",
#         "image_encoder.transformer.resblocks.11",
#     ]

#     for key, value in model.named_parameters():
#         if any(k in key for k in visual_open_keywords):
#             value.requires_grad_(True)
#             params += [{
#                 "params": [value],
#                 "lr": visual_lr,
#                 "weight_decay": cfg.SOLVER.STAGE1.WEIGHT_DECAY
#             }]
#             keys += [f"{key} | lr={visual_lr}"]

#     if len(params) == 0:
#         raise RuntimeError("No trainable parameters found for stage1")

#     print("Stage1 trainable params:")
#     for key in keys:
#         print("  ", key)

#     if cfg.SOLVER.STAGE1.OPTIMIZER_NAME == 'SGD':
#         optimizer = getattr(torch.optim, cfg.SOLVER.STAGE1.OPTIMIZER_NAME)(
#             params, momentum=cfg.SOLVER.STAGE1.MOMENTUM
#         )
#     elif cfg.SOLVER.STAGE1.OPTIMIZER_NAME == 'AdamW':
#         optimizer = torch.optim.AdamW(
#             params,
#             lr=cfg.SOLVER.STAGE1.BASE_LR,
#             weight_decay=cfg.SOLVER.STAGE1.WEIGHT_DECAY
#         )
#     else:
#         optimizer = getattr(torch.optim, cfg.SOLVER.STAGE1.OPTIMIZER_NAME)(params)

#     return optimizer

def make_optimizer_2stage(cfg, model, center_criterion):
    for _, value in model.named_parameters():
        value.requires_grad_(True)
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "text_encoder" in key:
            value.requires_grad_(False)
            continue
        # freeze `prompt_learner` but keep `inversion_prompt_learner` trainable
        if "prompt_learner" in key and "inversion_prompt_learner" not in key:
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.STAGE2.BASE_LR
        weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.STAGE2.BASE_LR * cfg.SOLVER.STAGE2.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.STAGE2.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')
        
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]
    if cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.STAGE2.MOMENTUM)
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE2.BASE_LR, weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.STAGE2.CENTER_LR)

    return optimizer, optimizer_center
