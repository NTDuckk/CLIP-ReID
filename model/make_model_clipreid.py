import torch
import torch.nn as nn
import numpy as np
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


class IM2TEXT(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=3, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)


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

class InversionPromptLearner3(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.prompt_clothes = IM2TEXT(
            embed_dim=clip_model.visual.output_dim,
            middle_dim=512,
            output_dim=clip_model.transformer.width,
            n_layer=3,
            dropout=0.1
        )
        self.prompt_hairstyle = IM2TEXT(
            embed_dim=clip_model.visual.output_dim,
            middle_dim=512,
            output_dim=clip_model.transformer.width,
            n_layer=3,
            dropout=0.1
        )
        self.prompt_carrying = IM2TEXT(
            embed_dim=clip_model.visual.output_dim,
            middle_dim=512,
            output_dim=clip_model.transformer.width,
            n_layer=3,
            dropout=0.1
        )

    def forward(self, image_feature):
        clothes_token = self.prompt_clothes(image_feature).unsqueeze(1)
        hairstyle_token = self.prompt_hairstyle(image_feature).unsqueeze(1)
        carrying_token = self.prompt_carrying(image_feature).unsqueeze(1)
        return clothes_token, hairstyle_token, carrying_token

class InversionPromptLearner5(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.prompt_top = IM2TEXT(
            embed_dim=clip_model.visual.output_dim,
            middle_dim=512,
            output_dim=clip_model.transformer.width,
            n_layer=3,
            dropout=0.1
        )
        self.prompt_underneath = IM2TEXT(
            embed_dim=clip_model.visual.output_dim,
            middle_dim=512,
            output_dim=clip_model.transformer.width,
            n_layer=3,
            dropout=0.1
        )
        self.prompt_shoes = IM2TEXT(
            embed_dim=clip_model.visual.output_dim,
            middle_dim=512,
            output_dim=clip_model.transformer.width,
            n_layer=3,
            dropout=0.1
        )
        self.prompt_hairstyle = IM2TEXT(
            embed_dim=clip_model.visual.output_dim,
            middle_dim=512,
            output_dim=clip_model.transformer.width,
            n_layer=3,
            dropout=0.1
        )
        self.prompt_carrying = IM2TEXT(
            embed_dim=clip_model.visual.output_dim,
            middle_dim=512,
            output_dim=clip_model.transformer.width,
            n_layer=3,
            dropout=0.1
        )
        
    def forward(self, image_feature):
        top_token = self.prompt_top(image_feature).unsqueeze(1)
        underneath_token = self.prompt_underneath(image_feature).unsqueeze(1)
        shoes_token = self.prompt_shoes(image_feature).unsqueeze(1)
        hairstyle_token = self.prompt_hairstyle(image_feature).unsqueeze(1)
        carrying_token = self.prompt_carrying(image_feature).unsqueeze(1)
        return top_token, underneath_token, shoes_token, hairstyle_token, carrying_token

class Prompt_Cat3(nn.Module):
    """
    Prompt learner for 3 attributes: clothes, hairstyle, carrying.
    Uses a fixed sentence with 3 X placeholders, and inserts attribute tokens at those positions.
    """
    def __init__(self, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a vehicle wearing X clothes, having X hairstyle and carrying X."
        else:
            ctx_init = "A photo of a person wearing X clothes, having X hairstyle and carrying X."

        ctx_dim = 512
        ctx_init = ctx_init.replace("_", " ")
        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts

        # Find positions of X tokens (token id 343 in CLIP)
        x_token_id = 343
        token_ids = tokenized_prompts[0].tolist()
        x_positions = [i for i, t in enumerate(token_ids) if t == x_token_id]
        if len(x_positions) != 3:
            raise RuntimeError(f"Expected 3 X tokens, found {len(x_positions)} at {x_positions}")

        # Register buffer segments
        self.register_buffer("token_prefix_0", embedding[:, :x_positions[0], :])        # before first X
        self.register_buffer("token_prefix_1", embedding[:, x_positions[0]+1:x_positions[1], :])  # between X1 and X2
        self.register_buffer("token_prefix_2", embedding[:, x_positions[1]+1:x_positions[2], :])  # between X2 and X3
        self.register_buffer("token_suffix", embedding[:, x_positions[2]+1:, :])                 # after last X

    def forward(self, label, prom_list):
        # prom_list: list of 3 tensors, each shape [B, 1, D] or [B, D]
        if prom_list is None or len(prom_list) != 3:
            raise ValueError("prom_list must be a list/tuple of length 3")

        b = label.shape[0]
        formatted = []
        for i, token in enumerate(prom_list):
            if token.dim() == 2:
                token = token.unsqueeze(1)
            elif token.dim() != 3:
                raise ValueError(f"prom_list[{i}] must be 2D or 3D, got {token.dim()}D")
            if token.size(2) != self.token_prefix_0.size(2):
                raise ValueError(f"prom_list[{i}] last dim must be {self.token_prefix_0.size(2)}")
            formatted.append(token.to(device=self.token_prefix_0.device, dtype=self.token_prefix_0.dtype))

        # Build prompt sequence
        pieces = [self.token_prefix_0.expand(b, -1, -1)]
        for i, token in enumerate(formatted):
            pieces.append(token)
            if i < len(formatted) - 1:
                pieces.append(getattr(self, f"token_prefix_{i+1}").expand(b, -1, -1))
        pieces.append(self.token_suffix.expand(b, -1, -1))

        prompts = torch.cat(pieces, dim=1)
        return prompts

# class Prompt_Cat5(nn.Module):
#     """
#     Prompt learner for 5 attributes: top, underneath, shoes, hairstyle, carrying.
#     Uses a fixed sentence with 5 X placeholders, and inserts attribute tokens at those positions.
#     """
#     def __init__(self, dataset_name, dtype, token_embedding):
#         super().__init__()
#         if dataset_name == "VehicleID" or dataset_name == "veri":
#             ctx_init = "A photo of a vehicle wearing X on top, X underneath, X shoes, having X hairstyle and carrying X."
#         else:
#             ctx_init = "A photo of a person wearing X on top, X underneath, X shoes, having X hairstyle and carrying X."

#         ctx_dim = 512
#         tokenized_prompts = clip.tokenize(ctx_init).cuda()
#         with torch.no_grad():
#             embedding = token_embedding(tokenized_prompts).type(dtype)
#         self.tokenized_prompts = tokenized_prompts

#         # Find positions of X tokens
#         x_token_id = 343
#         token_ids = tokenized_prompts[0].tolist()
#         x_positions = [i for i, t in enumerate(token_ids) if t == x_token_id]
#         if len(x_positions) != 5:
#             raise RuntimeError(f"Expected 5 X tokens, found {len(x_positions)} at {x_positions}")

#         # Register buffer segments
#         self.register_buffer("token_prefix_0", embedding[:, :x_positions[0], :])        # before first X
#         self.register_buffer("token_prefix_1", embedding[:, x_positions[0]+1:x_positions[1], :])  # between X1 and X2
#         self.register_buffer("token_prefix_2", embedding[:, x_positions[1]+1:x_positions[2], :])  # between X2 and X3
#         self.register_buffer("token_prefix_3", embedding[:, x_positions[2]+1:x_positions[3], :])  # between X3 and X4
#         self.register_buffer("token_prefix_4", embedding[:, x_positions[3]+1:x_positions[4], :])  # between X4 and X5
#         self.register_buffer("token_suffix", embedding[:, x_positions[4]+1:, :])                 # after last X
#     def forward(self, label, prom_list):
#         # prom_list: list of 5 tensors, each shape [B, 1, D] or [B, D]
#         if prom_list is None or len(prom_list) != 5:
#             raise ValueError("prom_list must be a list/tuple of length 5")

#         b = label.shape[0]
#         formatted = []
#         for i, token in enumerate(prom_list):
#             if token.dim() == 2:
#                 token = token.unsqueeze(1)
#             elif token.dim() != 3:
#                 raise ValueError(f"prom_list[{i}] must be 2D or 3D, got {token.dim()}D")
#             if token.size(2) != self.token_prefix_0.size(2):
#                 raise ValueError(f"prom_list[{i}] last dim must be {self.token_prefix_0.size(2)}")
#             formatted.append(token.to(device=self.token_prefix_0.device, dtype=self.token_prefix_0.dtype))

#         # Build prompt sequence
#         pieces = [self.token_prefix_0.expand(b, -1, -1)]
#         for i, token in enumerate(formatted):
#             pieces.append(token)
#             if i < len(formatted) - 1:
#                 pieces.append(getattr(self, f"token_prefix_{i+1}").expand(b, -1, -1))
#         pieces.append(self.token_suffix.expand(b, -1, -1))

#         prompts = torch.cat(pieces, dim=1)
#         return prompts

#------ debug---------
# class Prompt_Cat5(nn.Module):
#     """
#     Prompt template with 5 X placeholders.
#     Insert 5 attribute tokens directly into the X positions.
#     """
#     def __init__(self, dataset_name, dtype, token_embedding):
#         super().__init__()
#         if dataset_name == "VehicleID" or dataset_name == "veri":
#             template = "A photo of a vehicle wearing X on top, X underneath, X shoes, having X hairstyle and carrying X."
#         else:
#             template = "A photo of a person wearing X on top, X underneath, X shoes, having X hairstyle and carrying X."

#         self.num_attributes = 5

#         tokenized_prompts = clip.tokenize(template).cuda()
#         with torch.no_grad():
#             embedding = token_embedding(tokenized_prompts).type(dtype)

#         self.tokenized_prompts = tokenized_prompts
#         self.register_buffer("template_embedding", embedding)   # [1, 77, ctx_dim]

#         x_token_id = clip.tokenize("X")[0, 1].item()
#         x_positions = (tokenized_prompts[0] == x_token_id).nonzero(as_tuple=False).view(-1)

#         if x_positions.shape[0] != self.num_attributes:
#             raise RuntimeError(
#                 f"Expected {self.num_attributes} X positions, got {x_positions.shape[0]}"
#             )

#         self.register_buffer("x_positions", x_positions)
#         self.dtype = dtype

#     def forward(self, prom_list):
#         if prom_list is None or len(prom_list) != self.num_attributes:
#             raise ValueError(f"prom_list must be a list/tuple of length {self.num_attributes}")

#         first = prom_list[0]
#         if first.dim() == 3:
#             B = first.shape[0]
#         elif first.dim() == 2:
#             B = first.shape[0]
#         else:
#             raise ValueError(f"prom_list[0] must be 2D or 3D, got {first.dim()}D")

#         prompts = self.template_embedding.expand(B, -1, -1).clone()

#         for i, token in enumerate(prom_list):
#             if token.dim() == 3:
#                 if token.shape[1] != 1:
#                     raise ValueError(f"prom_list[{i}] must have shape [B,1,D] if 3D, got {token.shape}")
#                 token = token.squeeze(1)   # [B,1,D] -> [B,D]
#             elif token.dim() != 2:
#                 raise ValueError(f"prom_list[{i}] must be 2D or 3D, got {token.dim()}D")

#             prompts[:, self.x_positions[i], :] = token.to(
#                 device=prompts.device,
#                 dtype=prompts.dtype
#             )

#         return prompts

class Prompt_Cat5(nn.Module):
    def __init__(self, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            template = "A photo of a vehicle wearing X on top, X underneath, X shoes, having X hairstyle and carrying X."
        else:
            template = "A photo of a person wearing X on top , X underneath , X hairstyle , X shoes , carrying X ."
                            
        self.num_attributes = 5

        tokenized_prompts = clip.tokenize(template).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)

        self.tokenized_prompts = tokenized_prompts
        self.register_buffer("template_embedding", embedding)   # [1, 77, D]

        x_token_id = clip.tokenize("X")[0, 1].item()
        x_positions = (tokenized_prompts[0] == x_token_id).nonzero(as_tuple=False).view(-1)

        self.register_buffer("x_positions", x_positions)
        self.dtype = dtype

    def forward(self, prom_list):
        B = prom_list[0].shape[0]
        prompts = self.template_embedding.expand(B, -1, -1).clone()

        for i in range(self.num_attributes):
            prompts[:, self.x_positions[i], :] = prom_list[i].squeeze(1).type(self.dtype)

        return prompts

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
        # ensure attribute exists even if SIE flags are disabled
        self.cv_embed = None
        self.att_flag = cfg.MODEL.ATT_FLAG

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
        if self.att_flag == 3:
            self.prompt_learner = Prompt_Cat3(dataset_name, clip_model.dtype, clip_model.token_embedding)
            self.inversion_prompt_learner = InversionPromptLearner3(clip_model)
        elif self.att_flag == 5:
            self.prompt_learner = Prompt_Cat5(dataset_name, clip_model.dtype, clip_model.token_embedding)
            self.inversion_prompt_learner = InversionPromptLearner5(clip_model)
        else:
            raise ValueError(f"att_flag must be 3 or 5, but got {self.att_flag}")

        self.text_encoder = TextEncoder(clip_model)

    def forward(
        self,
        x=None,
        label=None,
        get_image=False,
        get_text=False,
        get_text_inversion=False,
        image_features_for_inversion = None,
        cam_label=None,
        view_label=None,
        prom_list=None
    ):
        if get_text is True:
            prompts = self.prompt_learner(label, prom_list)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features

        if get_text_inversion == True:
            prom_list = list(self.inversion_prompt_learner(image_features_for_inversion))
            prompts = self.prompt_learner(prom_list)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)

            return text_features

        if get_image is True:
            if self.model_name == 'RN50':
                image_features_last, image_features, image_features_proj = self.image_encoder(x)
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                if self.cv_embed is not None and cam_label is not None and view_label is not None:
                    cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
                elif self.cv_embed is not None and cam_label is not None:
                    cv_embed = self.sie_coe * self.cv_embed[cam_label]
                elif self.cv_embed is not None and view_label is not None:
                    cv_embed = self.sie_coe * self.cv_embed[view_label]
                else:
                    cv_embed = None
                image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
                return image_features_proj[:, 0]

        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(
                image_features_last, image_features_last.shape[2:4]
            ).view(x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(
                image_features, image_features.shape[2:4]
            ).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if self.cv_embed is not None and cam_label is not None and view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif self.cv_embed is not None and cam_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif self.cv_embed is not None and view_label is not None:
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


# class _PromptCatBase(nn.Module):
#     def __init__(self, ctx_init, dtype, token_embedding, expected_x_count):
#         super().__init__()
#         self.ctx_dim = 512
#         self.dtype = dtype

#         ctx_init = ctx_init.replace('_', ' ')
#         tokenized_prompts = clip.tokenize(ctx_init).cuda()

#         with torch.no_grad():
#             embedding = token_embedding(tokenized_prompts).type(dtype)

#         self.tokenized_prompts = tokenized_prompts

#         x_token_ids = set()
#         x_token_ids.add(int(clip.tokenize('X')[0, 1].item()))
#         x_token_ids.add(int(clip.tokenize(' X')[0, 1].item()))

#         token_ids = tokenized_prompts[0].tolist()
#         x_pos = [i for i, t in enumerate(token_ids) if t in x_token_ids]

#         if len(x_pos) != expected_x_count:
#             raise RuntimeError(
#                 f'Expected exactly {expected_x_count} X tokens, but found {len(x_pos)} at positions {x_pos}'
#             )

#         self.register_buffer('token_prefix', embedding[:, :x_pos[0], :])
#         for idx in range(expected_x_count - 1):
#             self.register_buffer(
#                 f'token_mid_{idx + 1}',
#                 embedding[:, x_pos[idx] + 1:x_pos[idx + 1], :]
#             )
#         self.register_buffer('token_suffix', embedding[:, x_pos[-1] + 1:, :])
#         self.expected_x_count = expected_x_count

#     def _format_prompt_token(self, p, name):
#         if p.dim() == 2:
#             p = p.unsqueeze(1)  # [B, C] -> [B, 1, C]

#         if p.dim() != 3:
#             raise ValueError(f'{name} must have shape [B, C] or [B, 1, C], but got {tuple(p.shape)}')

#         if p.size(-1) != self.ctx_dim:
#             raise ValueError(f'{name} last dim must be {self.ctx_dim}, but got {p.size(-1)}')

#         return p.to(device=self.token_prefix.device, dtype=self.token_prefix.dtype)

# class Prompt_Cat3(_PromptCatBase):
#     def __init__(self, num_class, dataset_name, dtype, token_embedding):
#         if dataset_name == 'VehicleID' or dataset_name == 'veri':
#             ctx_init = 'A photo of a vehicle wearing X clothes, having X hairstyle and carrying X.'
#         else:
#             ctx_init = 'A photo of a person wearing X clothes, having X hairstyle and carrying X.'

#         super().__init__(ctx_init, dtype, token_embedding, expected_x_count=3)
#         self.num_class = num_class

#     def forward(self, label, prom_list):
#         # label is kept for compatibility with the existing interface
#         if prom_list is None:
#             raise ValueError('Prompt_Cat3 requires prom_list=[clothes_token, hairstyle_token, carrying_token]')

#         if not isinstance(prom_list, (list, tuple)) or len(prom_list) != 3:
#             raise ValueError('prom_list must be a list/tuple of length 3 for Prompt_Cat3')

#         tokens = [
#             self._format_prompt_token(prom_list[0], 'clothes_token'),
#             self._format_prompt_token(prom_list[1], 'hairstyle_token'),
#             self._format_prompt_token(prom_list[2], 'carrying_token'),
#         ]

#         b = tokens[0].shape[0]
#         pieces = [self.token_prefix.expand(b, -1, -1)]
#         for idx, token in enumerate(tokens, start=1):
#             pieces.append(token)
#             if idx < len(tokens):
#                 pieces.append(getattr(self, f'token_mid_{idx}').expand(b, -1, -1))
#         pieces.append(self.token_suffix.expand(b, -1, -1))

#         prompts = torch.cat(pieces, dim=1)
#         return prompts

# class Prompt_Cat5(_PromptCatBase):
#     def __init__(self, num_class, dataset_name, dtype, token_embedding):
#         # Use the exact prompt requested by the user.
#         ctx_init = 'A photo of a person wearing X on top, X underneath, X shoes, having X hairstyle and carrying X.'
#         super().__init__(ctx_init, dtype, token_embedding, expected_x_count=5)
#         self.num_class = num_class

#     def forward(self, label, prom_list):
#         # label is kept for compatibility with the existing interface
#         if prom_list is None:
#             raise ValueError(
#                 'Prompt_Cat5 requires prom_list=[top_token, underneath_token, shoes_token, hairstyle_token, carrying_token]'
#             )

#         if not isinstance(prom_list, (list, tuple)) or len(prom_list) != 5:
#             raise ValueError('prom_list must be a list/tuple of length 5 for Prompt_Cat5')

#         token_names = ['top_token', 'underneath_token', 'shoes_token', 'hairstyle_token', 'carrying_token']
#         tokens = [self._format_prompt_token(token, name) for token, name in zip(prom_list, token_names)]

#         b = tokens[0].shape[0]
#         pieces = [self.token_prefix.expand(b, -1, -1)]
#         for idx, token in enumerate(tokens, start=1):
#             pieces.append(token)
#             if idx < len(tokens):
#                 pieces.append(getattr(self, f'token_mid_{idx}').expand(b, -1, -1))
#         pieces.append(self.token_suffix.expand(b, -1, -1))

#         prompts = torch.cat(pieces, dim=1)
#         return prompts


# # Backward-compatible aliases (optional but helps older imports keep working)
# InversionPromptLearner = InversionPromptLearner3
# Prompt_Cat = Prompt_Cat3
