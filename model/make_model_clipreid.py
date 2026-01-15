import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .clip import clip
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
        if tokenized_prompts.size(0) == 1 and prompts.size(0) > 1:
            tokenized_prompts = tokenized_prompts.expand(prompts.size(0), -1)

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        device = x.device
        batch_indices = torch.arange(x.shape[0], device=device)
        index = tokenized_prompts.argmax(dim=-1).to(device)
        x = x[batch_indices, index] @ self.text_projection
        return x


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

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]

        clip_model = load_clip_to_cpu(
            self.model_name,
            self.h_resolution,
            self.w_resolution,
            self.vision_stride_size
        )
        clip_model.to("cuda")
        self.logit_scale = clip_model.logit_scale
        self.logit_scale.requires_grad_(False)
        self.image_encoder = clip_model.visual

        # SIE
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

        # PromptLearner đã được sửa theo logic mới:
        # - base/full như cũ
        # - thêm shape_ctx + build_shape_student_prompts + build_shape_teacher_prompts
        self.prompt_learner = PromptLearner(
            num_classes,
            dataset_name,
            clip_model.dtype,
            clip_model.token_embedding
        )

        self.text_encoder = TextEncoder(clip_model)

    # ============================================================
    # NEW: APIs cho Stage15 shape-distill (no label)
    # ============================================================
    def get_shape_student_text_features(self, labels):
        """
        Student prompt: " ... X X X X ... V V."
        returns: [B, D]
        """
        prompts, tok = self.prompt_learner.build_shape_student_prompts(labels)
        tok = tok.to(prompts.device)
        return self.text_encoder(prompts, tok)

    def get_shape_teacher_text_features(self, labels):
        """
        Teacher bank: 32 prompts " ... X X X X ... {shape}."
        returns: [B, 32, D]
        """
        prompts, tok = self.prompt_learner.build_shape_teacher_prompts(labels)  # [B,32,77,dim], [B,32,77]
        B, K, L, D = prompts.shape
        prompts = prompts.reshape(B * K, L, D)
        tok = tok.reshape(B * K, -1).to(prompts.device)
        feat = self.text_encoder(prompts, tok).view(B, K, -1)
        return feat

    # ============================================================
    # forward như cũ: Stage1/Stage2 dùng get_text/get_image
    # Stage15 shape-distill sẽ gọi 2 method ở trên trực tiếp
    # ============================================================
    def forward(self, x=None, label=None, get_image=False, get_text=False, cam_label=None, view_label=None):
        # ----- get_text: dùng prompt_learner base/full hiện tại -----
        if get_text and label is not None:
            prompts = self.prompt_learner(label)  # [B,77,dim]
            tokenized = self.prompt_learner.get_tokenized_prompts().to(prompts.device)  # [B,77] hoặc [1,77] expand trong TextEncoder
            text_features = self.text_encoder(prompts, tokenized)
            return text_features

        # ----- get_image: trả image_feature_proj CLS như code gốc -----
        if get_image is True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:, 0]

        # ----- normal forward -----
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = F.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1)
            img_feature = F.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
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

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj
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

SHAPE_PAIRS = [
    ("Muscular", "Slender"),
    ("Broad-Shouldered", "Narrow-Shouldered"),
    ("Heavyset", "Petite"),
    ("Tall", "Short"),
    ("Long Legs", "Short Legs"),
    ("Long Torso", "Short Torso"),
    ("Curvy", "Angular"),
    ("Full-Figured", "Skinny"),
    ("Stocky", "Willowy"),
    ("Pear-Shaped", "Apple-Shaped"),
    ("Athletic", "Non-Athletic"),
    ("Fit", "Unfit"),
    ("Large-Breasted", "Small-Breasted"),
    ("Long-Armed", "Short-Armed"),
    ("Long-Necked", "Short-Necked"),
    ("High-Waisted", "Low-Waisted"),
]

def _flatten_shapes(pairs):
    out = []
    for a, b in pairs:
        out.append(a)
        out.append(b)
    return out  # len=32

def _placeholder_token_ids(ch: str):
    # robust: token id for "X" and " X" (tương tự cho "V")
    tid1 = clip.tokenize(ch)[0][1].item()
    tid2 = clip.tokenize(" " + ch)[0][1].item()
    return tid1, tid2

def _find_positions(token_row: torch.Tensor, token_ids, expected: int, name: str):
    # token_row: [77]
    mask = torch.zeros_like(token_row, dtype=torch.bool)
    for tid in token_ids:
        mask |= (token_row == tid)
    pos = torch.nonzero(mask, as_tuple=False).view(-1).tolist()
    if len(pos) != expected:
        raise ValueError(f"[PromptLearner] {name}: expected {expected} positions, got {len(pos)} ({pos})")
    return pos


class PromptLearner(nn.Module):
    """
    - base prompt (stage1):  "A photo of a X X X X person."
      -> learn cls_ctx (4 tokens / ID)

    - full prompt (stage15 old + stage2): 
        "A photo of a X X X X person who has X height and X body shape."
      -> learn attr_ctx (2 tokens / ID)  (giữ lại nếu bạn còn dùng)

    - NEW stage15 (shape distill, no label):
      Student prompt: "A photo of a X X X X person who is V V."
        -> learn shape_ctx (2 tokens / ID)
      Teacher bank: 32 prompts:
        "A photo of a X X X X person who is {shape}."
    """
    def __init__(self, num_class, dataset_name, dtype, token_embedding, shape_pairs=SHAPE_PAIRS):
        super().__init__()

        device = token_embedding.weight.device
        ctx_dim = token_embedding.weight.shape[1]

        self.dtype = dtype
        self.num_class = num_class

        # ---------- texts ----------
        if dataset_name in ["VehicleID", "veri"]:
            base_text = "A photo of a X X X X vehicle."
            full_text = "A photo of a X X X X vehicle."  # nếu vehicle không dùng attr
            shape_student_text = "A photo of a X X X X vehicle who is V V."
            shape_teacher_tpl  = "A photo of a X X X X vehicle who is {}."
        else:
            base_text = "A photo of a X X X X person."
            full_text = "A photo of a X X X X person who has X height and X body shape."
            shape_student_text = "A photo of a X X X X person who is V V."
            shape_teacher_tpl  = "A photo of a X X X X person who is {}."

        self.shapes = _flatten_shapes(shape_pairs)  # 32

        # ---------- placeholder ids ----------
        x_ids = _placeholder_token_ids("X")
        v_ids = _placeholder_token_ids("V")

        # ---------- learnable tokens ----------
        # cls_ctx: (num_class, 4, dim)
        self.n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, self.n_cls_ctx, ctx_dim, dtype=dtype, device=device)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # attr_ctx: (num_class, 2, dim)  (giữ lại nếu bạn còn dùng full prompt)
        self.n_attr_ctx = 2
        attr_vectors = torch.empty(num_class, self.n_attr_ctx, ctx_dim, dtype=dtype, device=device)
        nn.init.normal_(attr_vectors, std=0.02)
        self.attr_ctx = nn.Parameter(attr_vectors)

        # shape_ctx: (num_class, 2, dim)  (NEW)
        self.n_shape_ctx = 2
        shape_vectors = torch.empty(num_class, self.n_shape_ctx, ctx_dim, dtype=dtype, device=device)
        nn.init.normal_(shape_vectors, std=0.02)
        self.shape_ctx = nn.Parameter(shape_vectors)

        # ---------- BASE PROMPT (fixed parts) ----------
        tokenized_base = clip.tokenize(base_text).to(device)
        with torch.no_grad():
            embed_base = token_embedding(tokenized_base).type(dtype)

        x_pos_base = _find_positions(tokenized_base[0], x_ids, expected=self.n_cls_ctx, name="base X")
        self.register_buffer("tokenized_base", tokenized_base)
        self.register_buffer("prefix_base", embed_base[:, :x_pos_base[0], :])
        self.register_buffer("suffix_base", embed_base[:, x_pos_base[-1] + 1 :, :])

        # ---------- FULL PROMPT (fixed parts) ----------
        tokenized_full = clip.tokenize(full_text).to(device)
        with torch.no_grad():
            embed_full = token_embedding(tokenized_full).type(dtype)

        # full có 6 X: X1..X4 + X_height + X_shape
        x_pos_full = _find_positions(tokenized_full[0], x_ids, expected=6, name="full X")
        p0 = x_pos_full[0]
        p4 = x_pos_full[3]
        ph = x_pos_full[4]
        ps = x_pos_full[5]

        self.register_buffer("tokenized_full", tokenized_full)
        self.register_buffer("prefix_full", embed_full[:, :p0, :])
        self.register_buffer("mid_full_1", embed_full[:, p4 + 1 : ph, :])
        self.register_buffer("mid_full_2", embed_full[:, ph + 1 : ps, :])
        self.register_buffer("suffix_full", embed_full[:, ps + 1 :, :])

        # ---------- SHAPE STUDENT PROMPT (fixed parts) ----------
        tokenized_shape_student = clip.tokenize(shape_student_text).to(device)
        with torch.no_grad():
            embed_shape_student = token_embedding(tokenized_shape_student).type(dtype)

        x_pos_ss = _find_positions(tokenized_shape_student[0], x_ids, expected=self.n_cls_ctx, name="shape_student X")
        v_pos_ss = _find_positions(tokenized_shape_student[0], v_ids, expected=self.n_shape_ctx, name="shape_student V")
        if max(x_pos_ss) >= min(v_pos_ss):
            raise ValueError(f"[PromptLearner] shape_student: X positions must be before V positions. x={x_pos_ss}, v={v_pos_ss}")

        self.register_buffer("tokenized_shape_student", tokenized_shape_student)
        self.register_buffer("pre_shape_student", embed_shape_student[:, :x_pos_ss[0], :])
        self.register_buffer("mid_shape_student", embed_shape_student[:, x_pos_ss[-1] + 1 : v_pos_ss[0], :])
        self.register_buffer("suf_shape_student", embed_shape_student[:, v_pos_ss[-1] + 1 :, :])

        # ---------- SHAPE TEACHER BANK (32 prompts fixed parts) ----------
        teacher_texts = [shape_teacher_tpl.format(s) for s in self.shapes]
        tokenized_shape_teacher = clip.tokenize(teacher_texts).to(device)  # [32,77]
        with torch.no_grad():
            embed_shape_teacher = token_embedding(tokenized_shape_teacher).type(dtype)  # [32,77,dim]

        pre_list, suf_list = [], []
        for i in range(tokenized_shape_teacher.size(0)):
            x_pos_t = _find_positions(tokenized_shape_teacher[i], x_ids, expected=self.n_cls_ctx, name=f"shape_teacher[{i}] X")
            pre_list.append(embed_shape_teacher[i:i+1, :x_pos_t[0], :])          # [1,Lp,dim]
            suf_list.append(embed_shape_teacher[i:i+1, x_pos_t[-1] + 1 :, :])    # [1,Ls,dim]

        self.register_buffer("tokenized_shape_teacher", tokenized_shape_teacher)   # [32,77]
        self.register_buffer("pre_shape_teacher", torch.cat(pre_list, dim=0))     # [32,Lp,dim]
        self.register_buffer("suf_shape_teacher", torch.cat(suf_list, dim=0))     # [32,Ls,dim]

        # prompt mode: base / full (giữ như cũ)
        self.prompt_mode = "base"

    # --------- stage1/stage2 interface (giữ như cũ) ----------
    def set_prompt_mode(self, mode: str):
        assert mode in ["base", "full", "shape"]
        self.prompt_mode = mode

    def get_tokenized_prompts(self):
        if self.prompt_mode == "base":
            return self.tokenized_base
        elif self.prompt_mode == "full":
            return self.tokenized_full
        elif self.prompt_mode == "shape":
            return self.tokenized_shape_student
        else:
            raise ValueError(f"Unknown prompt_mode: {self.prompt_mode}")

    def forward(self, label):
        """
        Return prompt embeddings [B, 77, C] theo prompt_mode:

        - base  : "A photo of a X X X X person."
        - full  : "A photo of a X X X X person who has X height and X body shape."
                  (dùng attr_ctx: 2 token / ID)
        - shape : "A photo of a X X X X person who is V V."
                  (dùng shape_ctx: 2 token / ID, khớp với Stage15 student prompt)
        """
        if label is None:
            raise ValueError("PromptLearner.forward(label): label must not be None")

        if self.prompt_mode == "base":
            b = label.shape[0]
            ctx = self.cls_ctx[label]  # [B, 4, C]
            prefix = self.prefix_base.expand(b, -1, -1)
            suffix = self.suffix_base.expand(b, -1, -1)
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
            return prompts

        elif self.prompt_mode == "full":
            b = label.shape[0]
            ctx = self.cls_ctx[label]  # [B, 4, C]
            attr = self.attr_ctx[label]  # [B, 2, C]
            x_h = attr[:, 0:1, :]
            x_s = attr[:, 1:2, :]

            prefix = self.prefix_full.expand(b, -1, -1)
            mid1 = self.mid_full_1.expand(b, -1, -1)
            mid2 = self.mid_full_2.expand(b, -1, -1)
            suffix = self.suffix_full.expand(b, -1, -1)

            prompts = torch.cat([prefix, ctx, mid1, x_h, mid2, x_s, suffix], dim=1)
            return prompts

        elif self.prompt_mode == "shape":
            # dùng đúng template student của Stage15 để Stage2 có thể "ăn" shape_ctx
            prompts, _ = self.build_shape_student_prompts(label)
            return prompts
            

        else:
            raise ValueError(f"Unknown prompt_mode: {self.prompt_mode}")

    # --------- NEW APIs for stage15 shape distill ----------
    def build_shape_student_prompts(self, label):
        """
        Student: " ... X X X X ... V V."
        returns:
          prompts: [B,77,dim]
          tok:     [B,77]
        """
        b = label.shape[0]
        device = label.device

        cls_ctx = self.cls_ctx[label]      # [B,4,dim] (freeze in stage15)
        shp_ctx = self.shape_ctx[label]    # [B,2,dim] (learn)

        pre = self.pre_shape_student.to(device).expand(b, -1, -1)
        mid = self.mid_shape_student.to(device).expand(b, -1, -1)
        suf = self.suf_shape_student.to(device).expand(b, -1, -1)

        prompts = torch.cat([pre, cls_ctx, mid, shp_ctx, suf], dim=1)  # [B,77,dim]
        tok = self.tokenized_shape_student.to(device).expand(b, -1)    # [B,77]
        return prompts, tok

    def build_shape_teacher_prompts(self, label):
        """
        Teacher bank: 32 prompts " ... X X X X ... {shape}."
        Conditioned on cls_ctx[label] như logic bạn muốn.
        returns:
          prompts: [B,32,77,dim]
          tok:     [B,32,77]
        """
        b = label.shape[0]
        device = label.device

        cls_ctx = self.cls_ctx[label].unsqueeze(1).expand(b, 32, -1, -1)  # [B,32,4,dim]

        pre = self.pre_shape_teacher.to(device).unsqueeze(0).expand(b, -1, -1, -1)  # [B,32,Lp,dim]
        suf = self.suf_shape_teacher.to(device).unsqueeze(0).expand(b, -1, -1, -1)  # [B,32,Ls,dim]
        tok = self.tokenized_shape_teacher.to(device).unsqueeze(0).expand(b, -1, -1)  # [B,32,77]

        prompts = torch.cat([pre, cls_ctx, suf], dim=2)  # [B,32,77,dim]
        return prompts, tok