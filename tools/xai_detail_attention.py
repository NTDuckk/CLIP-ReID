import os
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from config import cfg
from model.make_model_clipreid import make_model


def load_checkpoint(model, weight_path):
    ckpt = torch.load(weight_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    return model


def build_transform(cfg):
    return transforms.Compose([
        transforms.Resize(cfg.INPUT.SIZE_TEST),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN,
            std=cfg.INPUT.PIXEL_STD
        )
    ])


def normalize_map(x):
    x = x - x.min()
    x = x / (x.max() + 1e-6)
    return x


def save_overlay(image_path, attn_map, save_path, alpha=0.45):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    h, w = img_np.shape[:2]
    attn_map = cv2.resize(attn_map, (w, h))
    attn_map = normalize_map(attn_map)

    plt.figure(figsize=(4, 8))
    plt.imshow(img_np)
    plt.imshow(attn_map, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


@torch.no_grad()
def visualize_detail_attention(cfg, model, image_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda"
    model.eval()
    model.to(device)

    transform = build_transform(cfg)

    pil_img = Image.open(image_path).convert("RGB")
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Nếu không dùng SIE camera/view thì để None
    cv_embed = None

    with torch.cuda.amp.autocast(enabled=True):
        image_features_last, image_features, image_features_proj = model.image_encoder(
            img_tensor,
            cv_embed
        )

        output = model.inversion_prompt_learner(
            image_features_proj,
            return_attn=True
        )

    tokens, attn_dict = output

    grid_h = model.h_resolution
    grid_w = model.w_resolution

    print("Patch grid:", grid_h, grid_w)

    for branch_name, attn_list in attn_dict.items():
        # Lấy attention của block cuối cùng
        attn = attn_list[-1]

        # attn shape: [B, heads, n_queries, N_patch]
        attn = attn[0]                       # [heads, n_queries, N_patch]
        attn = attn.mean(dim=0)              # [n_queries, N_patch]
        attn = attn.mean(dim=0)              # [N_patch]

        attn = attn.float().detach().cpu().numpy()
        attn = attn.reshape(grid_h, grid_w)

        save_path = os.path.join(save_dir, f"{branch_name}_attention.png")
        save_overlay(image_path, attn, save_path)

        print(f"Saved {branch_name} attention map to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--weight", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--camera_num", type=int, default=0)
    parser.add_argument("--view_num", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="xai_outputs")

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    model = make_model(
        cfg,
        num_class=args.num_classes,
        camera_num=args.camera_num,
        view_num=args.view_num
    )

    model = load_checkpoint(model, args.weight)

    visualize_detail_attention(
        cfg=cfg,
        model=model,
        image_path=args.image,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()