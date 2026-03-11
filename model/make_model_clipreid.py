import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts): 
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1).to(x.device)] @ self.text_projection 
        return x

class ShapeGuidedAttention(nn.Module):
    """Cross-attention: body shape text queries attend to image patch tokens."""
    def __init__(self, vis_dim=768, text_dim=512, num_pairs=16, out_dim=256):
        super().__init__()
        self.num_pairs = num_pairs
        self.text_proj = nn.Linear(text_dim, vis_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=vis_dim, num_heads=8, batch_first=True
        )
        self.shape_head = nn.Sequential(
            nn.Linear(vis_dim * num_pairs, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, patch_tokens, shape_text_feats):
        """
        patch_tokens:     [B, num_patches, vis_dim]
        shape_text_feats: [32, text_dim]  (16 pairs, frozen)
        Returns: shape_feat [B, out_dim], attn_weights [B, 16, num_patches]
        """
        B = patch_tokens.shape[0]
        queries = []
        for i in range(self.num_pairs):
            q = shape_text_feats[2 * i] - shape_text_feats[2 * i + 1]
            queries.append(q)
        queries = torch.stack(queries)                              # [16, text_dim]
        queries = self.text_proj(queries)                           # [16, vis_dim]
        queries = queries.unsqueeze(0).expand(B, -1, -1)            # [B, 16, vis_dim]

        attn_out, attn_weights = self.cross_attn(
            query=queries, key=patch_tokens, value=patch_tokens
        )  # attn_out: [B, 16, vis_dim]

        attn_flat = attn_out.reshape(B, -1)                         # [B, 16*vis_dim]
        shape_feat = self.shape_head(attn_flat)                     # [B, out_dim]
        return shape_feat, attn_weights


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE   

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        self.use_shape = (dataset_name not in ["VehicleID", "veri"])
        shape_dim = cfg.MODEL.SHAPE_DIM  # default 256

        # --- Shape-Aware components (PP4) ---
        if self.use_shape:
            with torch.no_grad():
                shape_text_feats = self._compute_shape_text_feats(clip_model)
            self.register_buffer('shape_text_feats', shape_text_feats)  # [32, 512]

            if self.model_name == 'ViT-B-16':
                self.shape_guided_attn = ShapeGuidedAttention(
                    vis_dim=self.in_planes, text_dim=self.in_planes_proj,
                    num_pairs=16, out_dim=shape_dim
                )
            else:  # RN50 fallback: simple MLP on bipolar scores
                self.shape_mlp = nn.Sequential(
                    nn.Linear(16, 128), nn.ReLU(), nn.Linear(128, shape_dim)
                )

            self.classifier_shape = nn.Linear(shape_dim, self.num_classes, bias=False)
            self.classifier_shape.apply(weights_init_classifier)
            self.bottleneck_shape = nn.BatchNorm1d(shape_dim)
            self.bottleneck_shape.bias.requires_grad_(False)
            self.bottleneck_shape.apply(weights_init_kaiming)

        self.prompt_learner = PromptLearner(
            num_classes, dataset_name, clip_model.dtype,
            clip_model.token_embedding, use_shape=self.use_shape
        )
        self.text_encoder = TextEncoder(clip_model)

    @torch.no_grad()
    def _compute_shape_text_feats(self, clip_model):
        """Encode 32 body-shape prompt sentences via frozen CLIP text encoder."""
        shape_prompts = [
            "A photo of a muscular person",       "A photo of a slender person",
            "A photo of a broad-shouldered person","A photo of a narrow-shouldered person",
            "A photo of a heavyset person",        "A photo of a petite person",
            "A photo of a tall person",            "A photo of a short person",
            "A photo of a short-legged person",    "A photo of a long-legged person",
            "A photo of a long-torsoed person",    "A photo of a short-torsoed person",
            "A photo of a curvy person",           "A photo of a angular person",
            "A photo of a full-figured person",    "A photo of a skinny person",
            "A photo of a stocky person",          "A photo of a willowy person",
            "A photo of a pear-shaped person",     "A photo of a apple-shaped person",
            "A photo of a athletic person",        "A photo of a non-athletic person",
            "A photo of a fit person",             "A photo of a unfit person",
            "A photo of a large-breasted person",  "A photo of a small-breasted person",
            "A photo of a long-armed person",      "A photo of a short-armed person",
            "A photo of a long-necked person",     "A photo of a short-necked person",
            "A photo of a high-waisted person",    "A photo of a low-waisted person",
        ]
        tokens = clip.tokenize(shape_prompts).cuda()
        feats = clip_model.encode_text(tokens)           # [32, 512]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float()

    def compute_bipolar_scores(self, img_feat_proj):
        """Compute 16-dim bipolar body-shape scores via CLIP zero-shot."""
        img_norm = F.normalize(img_feat_proj.float(), dim=-1)
        text_norm = F.normalize(self.shape_text_feats.float(), dim=-1)
        scores = img_norm @ text_norm.t()                # [B, 32]
        bipolar = scores[:, 0::2] - scores[:, 1::2]     # [B, 16]
        return bipolar

    def forward(self, x=None, label=None, get_image=False, get_text=False,
                cam_label=None, view_label=None, bipolar_scores=None):
        if get_text:
            prompts = self.prompt_learner(label, bipolar_scores=bipolar_scores)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features

        if get_image:
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:, 0]

        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label is not None and view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        # --- Shape branch ---
        if self.use_shape:
            bp_scores = self.compute_bipolar_scores(img_feature_proj)
            if self.model_name == 'ViT-B-16':
                patch_tokens = image_features[:, 1:]            # [B, num_patches, 768]
                shape_feat, shape_attn = self.shape_guided_attn(
                    patch_tokens, self.shape_text_feats
                )
            else:  # RN50
                shape_feat = self.shape_mlp(bp_scores)
            shape_feat_bn = self.bottleneck_shape(shape_feat)
        else:
            bp_scores = None
            shape_feat = None
            shape_feat_bn = None

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            if self.use_shape:
                cls_score_shape = self.classifier_shape(shape_feat_bn)
                return (
                    [cls_score, cls_score_proj],
                    [img_feature_last, img_feature, img_feature_proj],
                    img_feature_proj,
                    cls_score_shape, shape_feat, bp_scores
                )
            else:
                return (
                    [cls_score, cls_score_proj],
                    [img_feature_last, img_feature, img_feature_proj],
                    img_feature_proj,
                    None, None, None
                )

        else:
            if self.use_shape:
                if self.neck_feat == 'after':
                    return torch.cat([feat, feat_proj, shape_feat_bn], dim=1)
                else:
                    return torch.cat([img_feature, img_feature_proj, shape_feat], dim=1)
            else:
                if self.neck_feat == 'after':
                    return torch.cat([feat, feat_proj], dim=1)
                else:
                    return torch.cat([img_feature, img_feature_proj], dim=1)


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding, use_shape=True):
        super().__init__()
        self.use_shape = use_shape

        if dataset_name == "VehicleID" or dataset_name == "veri":
            base_word = "vehicle"
        else:
            base_word = "person"

        n_cls_ctx = 4
        n_shape_ctx = 1 if use_shape else 0
        n_total_x = n_cls_ctx + n_shape_ctx
        x_tokens = " ".join(["X"] * n_total_x)
        ctx_init = f"A photo of a {x_tokens} {base_word}."

        ctx_dim = 512
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4  # number of fixed context words before learnable tokens

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts

        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # prefix = [SOS, A, photo, of, a]  (n_ctx + 1 = 5 tokens)
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        # suffix = tokens after all X placeholders (base_word, ., EOT, PAD...)
        self.register_buffer(
            "token_suffix",
            embedding[:, n_ctx + 1 + n_total_x:, :]
        )
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx
        self.n_shape_ctx = n_shape_ctx

        # --- Shape token components ---
        if use_shape:
            shape_words_pos = [
                "muscular", "broad-shouldered", "heavyset", "tall",
                "short-legged", "long-torsoed", "curvy", "full-figured",
                "stocky", "pear-shaped", "athletic", "fit",
                "large-breasted", "long-armed", "long-necked", "high-waisted",
            ]
            shape_words_neg = [
                "slender", "narrow-shouldered", "petite", "short",
                "long-legged", "short-torsoed", "angular", "skinny",
                "willowy", "apple-shaped", "non-athletic", "unfit",
                "small-breasted", "short-armed", "short-necked", "low-waisted",
            ]
            pos_embeds = self._encode_words(shape_words_pos, token_embedding, dtype)
            neg_embeds = self._encode_words(shape_words_neg, token_embedding, dtype)
            self.register_buffer('pos_embeds', pos_embeds)   # [16, 512]
            self.register_buffer('neg_embeds', neg_embeds)   # [16, 512]
            self.shape_gate = nn.Sequential(
                nn.Linear(16, 16), nn.Sigmoid()
            )

    @staticmethod
    def _encode_words(words, token_embedding, dtype):
        """Average token embeddings for each word/phrase."""
        device = token_embedding.weight.device
        word_embeds = []
        for word in words:
            tokens = clip.tokenize(word).to(device)
            with torch.no_grad():
                emb = token_embedding(tokens).type(dtype)   # [1, 77, 512]
            eot_pos = tokens[0].argmax().item()
            word_emb = emb[0, 1:eot_pos].mean(dim=0)       # [512]
            word_embeds.append(word_emb)
        return torch.stack(word_embeds)                     # [N, 512]

    def forward(self, label, bipolar_scores=None):
        cls_ctx = self.cls_ctx[label]                       # [B, 4, 512]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        if self.use_shape:
            if bipolar_scores is not None:
                gate = self.shape_gate(bipolar_scores.float())     # [B, 16]
                weighted = bipolar_scores.float() * gate           # [B, 16]
                w_pos = torch.clamp(weighted, min=0)               # [B, 16]
                w_neg = torch.clamp(-weighted, min=0)              # [B, 16]
                shape_token = (
                    w_pos.unsqueeze(-1) * self.pos_embeds.unsqueeze(0)
                    + w_neg.unsqueeze(-1) * self.neg_embeds.unsqueeze(0)
                )                                                  # [B, 16, 512]
                shape_token = shape_token.sum(dim=1, keepdim=True)  # [B, 1, 512]
                shape_token = shape_token / (shape_token.norm(dim=-1, keepdim=True) + 1e-8)
            else:
                shape_token = torch.zeros(
                    b, 1, 512, dtype=cls_ctx.dtype, device=cls_ctx.device
                )
            prompts = torch.cat([prefix, cls_ctx, shape_token, suffix], dim=1)
        else:
            prompts = torch.cat([prefix, cls_ctx, suffix], dim=1)

        return prompts

