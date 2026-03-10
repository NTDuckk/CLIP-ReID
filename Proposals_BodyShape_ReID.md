# Đề xuất phương pháp: Tích hợp Body Shape Attributes vào CLIP-ReID cho Long-term Person Re-ID

> **Bối cảnh**: Baseline = CLIP-ReID (2-stage, ViT-B/16). Mục tiêu: Thêm đặc trưng hình dáng cơ thể (cao/thấp, mập/ốm, tỷ lệ chi...) để tăng khả năng phân biệt cùng một người trong quan sát dài hạn (long-term ReID), khi quần áo, kiểu tóc có thể thay đổi nhưng hình dáng cơ thể tương đối ổn định.
> 
> **Ràng buộc quan trọng**: KHÔNG có annotation body shape trên ảnh. Chỉ có **16 cặp từ mô tả body shape** (bipolar adjectives). Tận dụng **CLIP zero-shot** để tạo pseudo-label/soft-label tự động.

---

## Dữ liệu Body Shape có sẵn: 16 cặp từ bipolar

| # | Cực dương (+) | Cực âm (-) | Ngữ nghĩa |
|---|---|---|---|
| 1 | Muscular | Slender | Cơ bắp vs Mảnh khảnh |
| 2 | Broad-Shouldered | Narrow-Shouldered | Vai rộng vs Vai hẹp |
| 3 | Heavyset | Petite | To con vs Nhỏ nhắn |
| 4 | Tall | Short | Cao vs Thấp |
| 5 | Short Legs | Long Legs | Chân ngắn vs Chân dài |
| 6 | Long Torso | Short Torso | Thân dài vs Thân ngắn |
| 7 | Curvy | Angular | Đường cong vs Góc cạnh |
| 8 | Full-Figured | Skinny | Đầy đặn vs Gầy |
| 9 | Stocky | Willowy | Thấp mập vs Cao mảnh |
| 10 | Pear-Shaped | Apple-Shaped | Dáng quả lê vs Dáng quả táo |
| 11 | Athletic | Non-Athletic | Thể thao vs Không thể thao |
| 12 | Fit | Unfit | Cân đối vs Không cân đối |
| 13 | Large-Breasted | Small-Breasted | Ngực lớn vs Ngực nhỏ |
| 14 | Long-Armed | Short-Armed | Tay dài vs Tay ngắn |
| 15 | Long-Necked | Short-Necked | Cổ dài vs Cổ ngắn |
| 16 | High-Waisted | Low-Waisted | Eo cao vs Eo thấp |

**Lợi thế lớn**: CLIP đã được train trên hàng trăm triệu cặp image-text → **đã hiểu sẵn** ngữ nghĩa của tất cả 32 từ này. Có thể dùng zero-shot để tạo soft body shape descriptor cho mỗi ảnh **mà không cần annotation**.

---

## Cơ chế cốt lõi: CLIP Zero-Shot Body Shape Scoring

Với mỗi ảnh, dùng CLIP tính similarity với 32 text prompts:
```python
# Tạo 32 text prompts
shape_texts = [
    "A photo of a muscular person",        # cặp 1+
    "A photo of a slender person",          # cặp 1-
    "A photo of a broad-shouldered person", # cặp 2+
    "A photo of a narrow-shouldered person",# cặp 2-
    ...  # tổng 32 prompts
]

# CLIP encode text (1 lần duy nhất, cache lại)
shape_text_feats = clip_text_encoder(tokenize(shape_texts))  # [32, 512]

# Cho mỗi ảnh, tính similarity
image_feat = clip_image_encoder(image)   # [B, 512]  (projected feature)
shape_scores = image_feat @ shape_text_feats.T  # [B, 32]

# Chuyển thành 16 chiều bipolar: score(+) - score(-)
shape_descriptor = []
for i in range(16):
    score = shape_scores[:, 2*i] - shape_scores[:, 2*i+1]  # dương→cực+, âm→cực-
    shape_descriptor.append(score)
shape_descriptor = torch.stack(shape_descriptor, dim=1)  # [B, 16]
```

→ Mỗi ảnh có **16-dim body shape descriptor** (continuous, có giá trị +/-), **hoàn toàn tự động, không cần annotation**.

---

## Tại sao Body Shape quan trọng cho Long-term ReID?

| Đặc trưng | Short-term | Long-term |
|---|---|---|
| Quần áo, màu sắc | ✅ Ổn định | ❌ Thay đổi |
| Kiểu tóc | ✅ Ổn định | ❌ Có thể đổi |
| **Hình dáng cơ thể** | ✅ Ổn định | ✅ **Ổn định** |
| Dáng đi (gait) | ✅ | ✅ Ổn định |
| Tỷ lệ chiều cao/vai | ✅ | ✅ Ổn định |

**Kết luận**: Body shape (chiều cao tương đối, tỷ lệ cơ thể, độ gầy/béo) là đặc trưng **bất biến theo thời gian**, rất phù hợp cho long-term ReID.

---

## Phân tích hạn chế của CLIP-ReID gốc

1. **ID-specific prompt chỉ cho training IDs**: `cls_ctx[num_classes × 4 × 512]` — mỗi class có 4 learnable token riêng, **không generalize được cho unseen IDs** khi inference.
2. **Prompt không chứa ngữ nghĩa tường minh**: Các token `[X1][X2][X3][X4]` chỉ là vector học được, không mang ý nghĩa ngữ nghĩa rõ ràng (không biết token nào mô tả gì).
3. **Text không dùng khi inference**: Stage 2 freeze text, chỉ dùng visual feature → bỏ phí khả năng cross-modal của CLIP.
4. **Không có cơ chế tách biệt identity vs. attribute**: Tất cả thông tin nén vào 1 CLS token.
5. **Không khai thác body shape**: Đặc trưng hình dáng cơ thể hoàn toàn bị bỏ qua, trong khi CLIP có khả năng hiểu các mô tả body shape.

---

## Tham khảo từ các paper

| Paper | Ý tưởng chính | Bài học cho chúng ta |
|---|---|---|
| **PromptSG** (CVPR 2024) | Inversion network: ảnh → pseudo-token S* → "A photo of a S* person" + Cross-attention text→patch | End-to-end, dùng text lúc inference, cross-attention guide visual attention |
| **IADT** (MM 2025) | Tách S* (identity) + A1*A2*...Ak* (attributes) → "A photo of a S* person with A1*A2*...Ak*" | **Decouple identity vs attributes** bằng orthogonal constraint |
| **FLaN-Net** (IJCAI 2025) | S* (subject) + A* (attributes) + O* (occlusion) → cross-attention + dynamic fusion | Multi-type tokens, dynamic weight fusion, xử lý tốt occlusion |
| **AP-Attack** (ICCV 2025) | Prompt template chia attribute: S1(top), S2(bottom), S3(hair), S4(shoes), S5(bag) | **Chia body part cụ thể** cho từng thuộc tính |

---

## Đề xuất các phương pháp khả thi (CẬP NHẬT: dùng 16 cặp từ + CLIP zero-shot)

---

### Phương pháp 1: Shape-Described Prompt Learning (Mở rộng trực tiếp CLIP-ReID)

**Ý tưởng**: Giữ nguyên 2-stage pipeline CLIP-ReID. Thêm body shape tokens vào prompt, **khởi tạo bằng CLIP text embedding** của 16 cặp từ body shape thay vì random.

```
CLIP-ReID gốc:  "A photo of a [X1][X2][X3][X4] person"
Đề xuất:         "A photo of a [X1][X2][X3][X4] [B1][B2] person"
                                  ↑ ID tokens      ↑ Body shape tokens (khởi tạo từ shape text embeddings)
```

**Chi tiết kỹ thuật**:

```python
class ShapeAwarePromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding, clip_model):
        super().__init__()
        # Giữ nguyên ID tokens như CLIP-ReID gốc
        ctx_init = "A photo of a X X X X X X person."  # thêm 2 slot cho body shape
        n_cls_ctx = 4   # 4 ID-specific tokens (giữ nguyên)
        n_shape_ctx = 2  # 2 body shape tokens (MỚI)
        
        # ID tokens - giống CLIP-ReID gốc
        cls_vectors = torch.empty(num_class, n_cls_ctx, 512, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)
        
        # Body Shape tokens - KHỞI TẠO BẰNG CLIP TEXT EMBEDDINGS
        # Lấy trung bình embedding của các cặp body shape words 
        shape_words = ["muscular", "slender", "tall", "short", 
                       "athletic", "heavyset", "broad-shouldered", ...]
        shape_init = self._init_shape_from_clip(shape_words, clip_model)  # [2, 512]
        # Mỗi ID sẽ có body shape tokens riêng (có thể share nếu muốn)
        shape_vectors = shape_init.unsqueeze(0).expand(num_class, -1, -1).clone()
        self.shape_ctx = nn.Parameter(shape_vectors)  # [num_class, 2, 512]
    
    def forward(self, label):
        cls_ctx = self.cls_ctx[label]       # [B, 4, 512]
        shape_ctx = self.shape_ctx[label]   # [B, 2, 512]
        # Ghép: prefix + cls_ctx + shape_ctx + suffix
        prompts = torch.cat([prefix, cls_ctx, shape_ctx, suffix], dim=1)
        return prompts
```

**Stage 1 bổ sung - Shape Consistency Loss**:
```python
# Ý tưởng: Cùng 1 person ID → shape tokens phải gần nhau
# Dùng CLIP zero-shot scores làm pseudo-target
with torch.no_grad():
    shape_pseudo = compute_shape_descriptor(image_features, shape_text_feats)  # [B, 16]
    # shape_pseudo: 16-dim bipolar scores từ CLIP zero-shot

# Loss: text shape tokens nên encode thông tin tương tự shape_pseudo
# Thêm vào Stage 1: loss = loss_i2t + loss_t2i + α * shape_consistency_loss
```

**Ưu điểm**: Thay đổi ít nhất, dễ implement, backward-compatible
**Nhược điểm**: Vẫn ID-specific (không generalize cho unseen ID)

**Độ khó triển khai**: ⭐ Dễ (chỉ sửa PromptLearner)

---

### Phương pháp 2: Zero-Shot Shape Descriptor + Dual-Branch (Thực tế nhất)

**Ý tưởng**: Giữ nguyên CLIP-ReID pipeline, thêm 1 **Body Shape Branch** song song. Branch này dùng **CLIP zero-shot scores** từ 16 cặp từ làm soft body shape representation, rồi học một projection tinh chỉnh.

```
                    ┌─── CLIP-ReID Branch (giữ nguyên) ──────→ feat_reid [1280-dim]
Image → ViT Encoder─┤
                    └─── Body Shape Branch (mới) ────────────→ feat_shape [256-dim]
                    
       16 cặp từ ──→ CLIP Text Encoder (frozen) ──→ shape_anchors [32, 512]
                    
Final feature = cat(feat_reid, feat_shape)  → [1536-dim]
```

**Chi tiết kỹ thuật**:

```python
# ===== BƯỚC 0: Tạo shape text anchors (1 lần duy nhất) =====
shape_pairs = [
    ("A photo of a muscular person",    "A photo of a slender person"),
    ("A photo of a broad-shouldered person", "A photo of a narrow-shouldered person"),
    ("A photo of a heavyset person",    "A photo of a petite person"),
    ("A photo of a tall person",        "A photo of a short person"),
    ("A photo of a person with short legs", "A photo of a person with long legs"),
    ("A photo of a person with a long torso", "A photo of a person with a short torso"),
    ("A photo of a curvy person",       "A photo of an angular person"),
    ("A photo of a full-figured person", "A photo of a skinny person"),
    ("A photo of a stocky person",      "A photo of a willowy person"),
    ("A photo of a pear-shaped person", "A photo of an apple-shaped person"),
    ("A photo of an athletic person",   "A photo of a non-athletic person"),
    ("A photo of a fit person",         "A photo of an unfit person"),
    ("A photo of a large-breasted person", "A photo of a small-breasted person"),
    ("A photo of a long-armed person",  "A photo of a short-armed person"),
    ("A photo of a long-necked person", "A photo of a short-necked person"),
    ("A photo of a high-waisted person","A photo of a low-waisted person"),
]
# Encode tất cả → [32, 512], frozen
shape_text_feats = clip_text_encode(all_32_prompts)  # cached, không train

class BodyShapeBranch(nn.Module):
    def __init__(self, proj_dim=512, shape_dim=256, num_pairs=16):
        super().__init__()
        self.num_pairs = num_pairs
        
        # Learnable refinement: tinh chỉnh raw CLIP scores thành shape features
        # Raw scores: [B, 32] → bipolar: [B, 16] → project: [B, 256]
        self.shape_proj = nn.Sequential(
            nn.Linear(num_pairs, 128),
            nn.ReLU(),
            nn.Linear(128, shape_dim),
        )
        self.shape_bn = nn.BatchNorm1d(shape_dim)
        self.shape_bn.bias.requires_grad_(False)
        
    def forward(self, image_proj_feat, shape_text_feats):
        """
        image_proj_feat: [B, 512] - CLIP projected image feature (đã có sẵn từ image encoder)
        shape_text_feats: [32, 512] - frozen CLIP text embeddings của 32 shape descriptions
        """
        # Bước 1: Tính similarity với 32 shape texts
        raw_scores = image_proj_feat @ shape_text_feats.t()  # [B, 32]
        
        # Bước 2: Bipolar encoding - lấy hiệu giữa 2 cực
        bipolar = []
        for i in range(self.num_pairs):
            bipolar.append(raw_scores[:, 2*i] - raw_scores[:, 2*i+1])
        bipolar = torch.stack(bipolar, dim=1)  # [B, 16]
        
        # Bước 3: Learnable projection để tinh chỉnh
        shape_feat = self.shape_proj(bipolar)   # [B, 256]
        shape_feat = self.shape_bn(shape_feat)
        return shape_feat, bipolar  # trả cả bipolar để tính consistency loss

# ===== Tích hợp vào build_transformer =====
class build_transformer(nn.Module):  # sửa đổi
    def __init__(self, ...):
        # ... giữ nguyên tất cả CLIP-ReID gốc ...
        self.shape_branch = BodyShapeBranch(shape_dim=256)
        self.shape_classifier = nn.Linear(256, num_classes, bias=False)
        
    def forward(self, x=None, label=None, ...):
        # ... CLIP-ReID forward giữ nguyên ...
        # img_feature_proj = image_features_proj[:,0]  # [B, 512]
        
        shape_feat, bipolar = self.shape_branch(img_feature_proj, self.shape_text_feats)
        
        if self.training:
            cls_score_shape = self.shape_classifier(shape_feat)
            return ([cls_score, cls_score_proj, cls_score_shape], 
                    [img_feature_last, img_feature, img_feature_proj, shape_feat],
                    img_feature_proj, bipolar)
        else:
            feat_final = torch.cat([feat, feat_proj, shape_feat], dim=1)  # 768+512+256=1536
            return feat_final
```

**Losses Stage 2 bổ sung**:
```python
# Loss gốc CLIP-ReID (giữ nguyên)
loss_original = 0.25 * ID_LOSS + 1.0 * TRI_LOSS + 1.0 * I2T_LOSS

# Loss mới cho shape branch
loss_shape_id = xent(cls_score_shape, target)           # Shape cũng phải ID-classifiable
loss_shape_tri = triplet(shape_feat, target)             # Shape trong cùng ID phải gần nhau

# Shape consistency: cùng person → bipolar descriptor phải nhất quán
# (tự động enforce bởi triplet loss trên shape_feat)

loss_total = loss_original + λ1 * loss_shape_id + λ2 * loss_shape_tri
# Suggest: λ1=0.25, λ2=0.5
```

**Ưu điểm**: 
- **Thay đổi tối thiểu**: Giữ nguyên CLIP-ReID, chỉ thêm 1 branch nhỏ
- **Không cần annotation**: Dùng CLIP zero-shot hoàn toàn tự động
- **Body shape interpretable**: 16 bipolar scores có thể visualize, phân tích
- **Dễ ablation**: Bỏ shape branch = quay lại CLIP-ReID gốc
- **Shape feature bất biến**: 16 cặp từ mô tả đặc trưng cơ thể ổn định theo thời gian

**Nhược điểm**: Shape scores phụ thuộc chất lượng CLIP zero-shot, có thể noisy

**Độ khó triển khai**: ⭐ Dễ nhất, chắc nhất

---

### Phương pháp 3: Shape-Guided Attention Reweighting (Khai thác sâu hơn ngữ nghĩa)

**Ý tưởng**: Dùng 16 cặp body shape text embeddings làm **attention queries** để reweight patch tokens trong ViT. Mỗi cặp từ "hỏi" image: "phần nào mang thông tin về tall/short? muscular/slender?..." → selective attention cho body shape regions.

```
                                      ┌→ shape_attn_1 (muscular vs slender)
Patch Tokens [128, 768] ─── Cross-Attention ←── Shape Text Queries [16, 512]
                                      ├→ shape_attn_2 (broad vs narrow shoulder)
                                      └→ ... (16 attention maps)
                                      ↓ aggregate
                                 shape_feat [256-dim]
```

**Chi tiết kỹ thuật**:

```python
class ShapeGuidedAttention(nn.Module):
    """Dùng body shape text embeddings làm queries để attend vào patch tokens"""
    def __init__(self, vis_dim=768, text_dim=512, num_pairs=16, out_dim=256):
        super().__init__()
        self.num_pairs = num_pairs
        
        # Project text vào visual space 
        self.text_proj = nn.Linear(text_dim, vis_dim)
        
        # Cross-attention: text queries attend to patch tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=vis_dim, num_heads=8, batch_first=True
        )
        
        # Aggregate: 16 attended features → compact shape descriptor
        self.shape_head = nn.Sequential(
            nn.Linear(vis_dim * num_pairs, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.bn = nn.BatchNorm1d(out_dim)
    
    def forward(self, patch_tokens, shape_text_pairs):
        """
        patch_tokens: [B, 128, 768] - all patch embeddings từ ViT layer 11
        shape_text_pairs: [32, 512]  - 16 cặp text embeddings (frozen)
        """
        B = patch_tokens.shape[0]
        
        # Tạo 16 bipolar queries: embed(+) - embed(-) cho mỗi cặp
        queries = []
        for i in range(self.num_pairs):
            q = shape_text_pairs[2*i] - shape_text_pairs[2*i+1]  # bipolar direction
            queries.append(q)
        queries = torch.stack(queries)                # [16, 512]
        queries = self.text_proj(queries)             # [16, 768]
        queries = queries.unsqueeze(0).expand(B,-1,-1) # [B, 16, 768]
        
        # Cross-attention: shape queries attend to image patches
        attn_out, attn_weights = self.cross_attn(
            query=queries,           # [B, 16, 768] - "hỏi" về từng body shape
            key=patch_tokens,        # [B, 128, 768] - patch tokens "trả lời"
            value=patch_tokens       # [B, 128, 768]
        )
        # attn_out: [B, 16, 768] - mỗi shape query thu được thông tin từ relevant patches
        # attn_weights: [B, 16, 128] - attention map cho mỗi shape query (visualizable!)
        
        # Aggregate thành compact feature
        attn_flat = attn_out.reshape(B, -1)            # [B, 16*768]
        shape_feat = self.shape_head(attn_flat)        # [B, 256]
        shape_feat = self.bn(shape_feat)
        return shape_feat, attn_weights

# CÁCH DÙNG: Chèn sau image_encoder, trước losses
# patch_tokens = image_features[:, 1:]  # bỏ CLS token
# shape_feat, shape_attn = self.shape_guided_attn(patch_tokens, shape_text_feats)
```

**Lợi ích attention maps**:
- `attn_weights[:, 0, :]` → patches nào cho biết muscular/slender (thường focus ngực, tay)
- `attn_weights[:, 3, :]` → patches nào cho biết tall/short (focus toàn thân)
- `attn_weights[:, 1, :]` → patches nào cho biết broad/narrow shoulder (focus vai)
- → **Interpretable + visualizable** trong paper

**Ưu điểm**: 
- Language-guided attention, khai thác cross-modal rất hiệu quả
- Attention maps visualizable → tốt cho paper analysis
- Mỗi shape dimension attend vào đúng body region liên quan

**Nhược điểm**: Thêm cross-attention → tăng computation, cần tune nhiều hơn

**Độ khó triển khai**: ⭐⭐ Trung bình

---

### Phương pháp 4: Shape-Aware Contrastive Prompt (Kết hợp PP2 + PP3, contribution mạnh nhất)

**Ý tưởng**: Kết hợp cả 3 yếu tố:
1. **CLIP zero-shot shape scoring** (16-dim bipolar) → soft pseudo-label
2. **Shape-guided cross-attention** → attend đúng body regions
3. **Shape-conditioned text prompt** → structured prompt tích hợp body shape

```
Prompt: "A photo of a [S*] [SHAPE_DESC] person"
         S*         = ID tokens (learnable, giống CLIP-ReID)
         SHAPE_DESC = weighted combination của body shape word embeddings
                      (trọng số = CLIP zero-shot bipolar scores)
```

**Chi tiết kỹ thuật**:

```python
class ShapeConditionedPromptLearner(nn.Module):
    def __init__(self, num_class, clip_model):
        super().__init__()
        # ID tokens giống CLIP-ReID
        self.cls_ctx = nn.Parameter(torch.randn(num_class, 4, 512))
        
        # Pre-compute shape word embeddings (FROZEN)
        shape_words_pos = ["muscular", "broad-shouldered", "heavyset", "tall", 
                          "short-legged", "long-torsoed", "curvy", "full-figured",
                          "stocky", "pear-shaped", "athletic", "fit",
                          "large-breasted", "long-armed", "long-necked", "high-waisted"]
        shape_words_neg = ["slender", "narrow-shouldered", "petite", "short",
                          "long-legged", "short-torsoed", "angular", "skinny", 
                          "willowy", "apple-shaped", "non-athletic", "unfit",
                          "small-breasted", "short-armed", "short-necked", "low-waisted"]
        
        # [16, 512] embeddings cho mỗi cực
        self.register_buffer('pos_embeds', encode_words(shape_words_pos, clip_model))
        self.register_buffer('neg_embeds', encode_words(shape_words_neg, clip_model))
        
        # Learnable: chọn top-K shape dimensions quan trọng nhất
        self.shape_gate = nn.Sequential(
            nn.Linear(16, 16), nn.Sigmoid()  # gate: shape nào active
        )
    
    def forward(self, label, bipolar_scores=None):
        """
        bipolar_scores: [B, 16] - từ CLIP zero-shot scoring (Stage 2 truyền vào)
        """
        cls_ctx = self.cls_ctx[label]  # [B, 4, 512]
        
        if bipolar_scores is not None:
            # Dynamic shape token: weighted sum của word embeddings
            # theo CLIP zero-shot scores
            gate = self.shape_gate(bipolar_scores)  # [B, 16] soft selection
            weighted_scores = bipolar_scores * gate  # [B, 16] gated scores
            
            # Khi score > 0: thiên về pos_embeds, score < 0: thiên về neg_embeds
            weights_pos = torch.clamp(weighted_scores, min=0)  # [B, 16]
            weights_neg = torch.clamp(-weighted_scores, min=0) # [B, 16]
            
            # Shape token = weighted sum (1 token tổng hợp)
            shape_token = (weights_pos.unsqueeze(-1) * self.pos_embeds.unsqueeze(0) +
                          weights_neg.unsqueeze(-1) * self.neg_embeds.unsqueeze(0))
            shape_token = shape_token.sum(dim=1, keepdim=True)  # [B, 1, 512]
            shape_token = shape_token / (shape_token.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            shape_token = torch.zeros(label.shape[0], 1, 512).to(label.device)
        
        # Ghép: prefix + cls_ctx + shape_token + suffix
        prompts = torch.cat([prefix, cls_ctx, shape_token, suffix], dim=1)
        return prompts
```

**Training pipeline (sửa đổi cả 2 stages)**:

```
Stage 1 (cải tiến):
  1. Extract image features (frozen) → img_feat_proj [N, 512]
  2. Compute bipolar_scores = zero_shot_shape(img_feat_proj, shape_texts) [N, 16]
  3. Train: PromptLearner(label, bipolar_scores) → text_features
  4. Loss = SupConLoss(img↔text) + α * ShapeConsistencyLoss

Stage 2 (cải tiến):
  1. Pre-compute text features WITH shape info → text_features [C, 512]
     (dùng mean bipolar_scores per class)
  2. Training loop: forward → 3 features + shape_feat
  3. Loss = ID + Triplet + I2T + β * ShapeTriplet + γ * ShapeID
  4. ShapeGuidedAttention reweight patches (optional, từ PP3)

Inference:
  Visual feature = cat(feat, feat_proj, shape_feat)  → 1536-dim
  (shape_feat có thể tính real-time từ img_feat_proj + shape_text_feats)
```

**Losses đầy đủ**:
```
L = L_ID_orig + L_Triplet_orig + L_I2T_orig          # CLIP-ReID gốc
  + λ1 * L_SupCon(shape-aware)                         # Shape-aware contrastive
  + λ2 * L_Triplet(shape_feat, target)                 # Shape feature discriminative  
  + λ3 * L_ShapeConsistency                            # Cùng person → shape gần nhau
```

`L_ShapeConsistency`: Trong mỗi batch, các ảnh cùng person ID phải có bipolar scores gần nhau:
```python
def shape_consistency_loss(bipolar, pids):
    loss = 0
    for pid in pids.unique():
        mask = (pids == pid)
        if mask.sum() > 1:
            shape_feats = bipolar[mask]  # [n, 16]
            mean_shape = shape_feats.mean(0)
            loss += ((shape_feats - mean_shape) ** 2).mean()
    return loss / pids.unique().shape[0]
```

**Ưu điểm**: 
- **Contribution mạnh nhất**: Kết hợp zero-shot + prompt learning + shape guidance
- Body shape có ngữ nghĩa tường minh (16 cặp từ có thể giải thích)
- Shape token dynamic theo từng ảnh (instance-specific thay vì ID-specific)
- Dùng được lúc inference cho unseen IDs

**Nhược điểm**: Phức tạp nhất, cần tune nhiều hyperparameters

**Độ khó triển khai**: ⭐⭐⭐ Khá phức tạp

---

### Phương pháp 5: Hierarchical Identity-Shape Tokenization (Kiểu IADT, paper-grade)

**Ý tưởng**: Kết hợp IADT + body shape, tách **3 loại semantics** vào orthogonal subspaces:
- **S\*** = Identity token (global feature → mapping network)
- **H\*** = Body shape token (**learned từ shape text anchors**)
- **A\*** = Appearance attribute tokens (local patches → learnable queries)

```
Prompt: "A photo of a [S*] person with [H1*][H2*] body and [A1*][A2*]...[Ak*]"
```

**Chi tiết kỹ thuật**:

```python
class HierarchicalTokenizer(nn.Module):
    def __init__(self, vis_dim=768, text_dim=512, num_shape_pairs=16, num_attr=4):
        super().__init__()
        
        # 1. Subject mapping (identity, giống PromptSG/IADT)
        self.subject_map = nn.Sequential(
            nn.Linear(vis_dim, text_dim), nn.ReLU(), 
            nn.Linear(text_dim, text_dim), nn.ReLU(),
            nn.Linear(text_dim, text_dim)
        )
        
        # 2. Body Shape mapping (MỚI - dùng CLIP shape text anchors)
        # Input = bipolar scores [16] → shape tokens [2, 512]
        self.shape_map = nn.Sequential(
            nn.Linear(num_shape_pairs, text_dim),
            nn.ReLU(),
            nn.Linear(text_dim, 2 * text_dim),  # output 2 tokens × 512
        )
        
        # 3. Appearance attribute mapping (giống IADT)
        self.attr_queries = nn.Parameter(torch.randn(num_attr, vis_dim))
        self.attr_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=vis_dim, nhead=8), num_layers=2
        )
        self.attr_map = nn.Sequential(
            nn.Linear(vis_dim, text_dim), nn.ReLU(), 
            nn.Linear(text_dim, text_dim)
        )
        
    def forward(self, global_feat, patch_feats, bipolar_scores):
        B = global_feat.shape[0]
        
        # Identity token
        S_star = self.subject_map(global_feat)  # [B, 512]
        
        # Body Shape tokens (từ CLIP zero-shot scores)
        shape_out = self.shape_map(bipolar_scores)           # [B, 1024]
        H_star = shape_out.reshape(B, 2, 512)                # [B, 2, 512]
        
        # Appearance tokens (từ patch features, giống IADT)
        queries = self.attr_queries.unsqueeze(0).expand(B,-1,-1)  # [B, 4, 768]
        combined = torch.cat([queries, patch_feats], dim=1)       # [B, 4+128, 768]
        attr_out = self.attr_transformer(combined)[:, :4]          # [B, 4, 768]
        A_star = self.attr_map(attr_out)                           # [B, 4, 512]
        
        return S_star, H_star, A_star

# Và prompt được ghép: "A photo of a [S*] person with [H1*][H2*] body and [A1*][A2*][A3*][A4*]"

# ORTHOGONAL CONSTRAINT (từ IADT):
# S*, H*, A* phải ở các subspace khác nhau
def orthogonal_loss(S, H, A):
    # Stack tất cả tokens
    all_tokens = torch.cat([S.unsqueeze(1), H, A], dim=1)  # [B, 1+2+4=7, 512]
    # Gram matrix
    G = torch.bmm(all_tokens, all_tokens.transpose(1,2))   # [B, 7, 7]
    I = torch.eye(7).unsqueeze(0).to(G.device)
    return ((G - I) ** 2).mean()
```

**Losses đầy đủ**:
```
L = L_SupCon(v↔text)                    # Image-text alignment
  + L_ID + L_Triplet                     # Standard ReID losses
  + λ1 * L_ortho(S*, H*, A*)            # Orthogonal constraint
  + λ2 * L_shape_consistency(H*, pids)   # Shape invariance
  + λ3 * L_shape_triplet(H*, pids)      # Shape discriminative
```

**Ưu điểm**: Novelty rất cao, tách biệt rõ 3 semantic spaces, dùng text-guided
**Nhược điểm**: Phức tạp nhất, cần end-to-end redesign

**Độ khó triển khai**: ⭐⭐⭐⭐ Phức tạp

---

## So sánh tổng hợp 5 phương pháp

| Phương pháp | Novelty | Dễ implement | Dùng 16 cặp từ | Body shape feature | Khuyến nghị |
|---|---|---|---|---|---|
| PP1: Shape Prompt Init | ⭐⭐ | ⭐⭐⭐⭐⭐ | Khởi tạo | ID-specific | Quick experiment |
| **PP2: Zero-Shot + Dual-Branch** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Core mechanism** | **16-dim bipolar→256** | **⭐ Bắt đầu từ đây** |
| PP3: Shape-Guided Attention | ⭐⭐⭐⭐ | ⭐⭐⭐ | Attention queries | Cross-attn features | Nâng cấp PP2 |
| **PP4: Shape-Aware Contrastive** | ⭐⭐⭐⭐ | ⭐⭐⭐ | **Full integration** | Dynamic prompt+feat | **⭐ Best overall** |
| PP5: Hierarchical IADT | ⭐⭐⭐⭐⭐ | ⭐⭐ | Mapping network | Orthogonal subspace | Research-grade |

---

## Chiến lược triển khai đề nghị cho KLTN

### Phase 1: Validation (1-2 tuần)
```
→ Implement PP2 (Zero-Shot + Dual-Branch)
→ Kiểm chứng: 16 bipolar scores có thực sự discriminative cho body shape?
→ Visualize bipolar scores cho các identity khác nhau
→ Chạy trên Market-1501, so sánh với CLIP-ReID baseline
```

### Phase 2: Enhancement (2-3 tuần)  
```
→ Nếu PP2 works → thêm PP3 (Shape-Guided Attention) lên trên
→ Hoặc nâng cấp thành PP4 (Shape-Aware Contrastive Prompt)
→ Ablation study: từng component đóng góp bao nhiêu
```

### Phase 3: Long-term Evaluation (1-2 tuần)
```
→ Chạy trên LTCC / PRCC (clothes-changing datasets)
→ Chứng minh body shape features bất biến khi thay đổi quần áo
→ Visualize attention maps (nếu dùng PP3)
```

---

## Cách dùng 16 cặp từ body shape hiệu quả

### 1. Full sentence prompts (khuyến nghị cho CLIP scoring)
```python
shape_prompts = [
    "A photo of a muscular person",
    "A photo of a slender person",
    "A photo of a broad-shouldered person",
    "A photo of a narrow-shouldered person",
    "A photo of a heavyset person",
    "A photo of a petite person",
    "A photo of a tall person",
    "A photo of a short person",
    "A photo of a person with short legs",
    "A photo of a person with long legs",
    "A photo of a person with a long torso",
    "A photo of a person with a short torso",
    "A photo of a curvy person",
    "A photo of an angular person",
    "A photo of a full-figured person",
    "A photo of a skinny person",
    "A photo of a stocky person",
    "A photo of a willowy person",
    "A photo of a pear-shaped person",
    "A photo of an apple-shaped person",
    "A photo of an athletic person",
    "A photo of a non-athletic person",
    "A photo of a fit person",
    "A photo of an unfit person",
    "A photo of a large-breasted person",
    "A photo of a small-breasted person",
    "A photo of a long-armed person",
    "A photo of a short-armed person",
    "A photo of a long-necked person",
    "A photo of a short-necked person",
    "A photo of a high-waisted person",
    "A photo of a low-waisted person",
]
```

### 2. Lọc bớt nếu cần (top-K relevant pairs)
Có thể lọc 16 → 8 pairs quan trọng nhất cho ReID:
- **Rất quan trọng**: Tall/Short, Athletic/Non-Athletic, Muscular/Slender, Heavyset/Petite
- **Quan trọng**: Broad/Narrow-Shouldered, Full-Figured/Skinny, Stocky/Willowy, Fit/Unfit
- **Ít quan trọng hơn** (khó thấy qua ảnh surveillance): Long/Short-Necked, High/Low-Waisted

### 3. Kết hợp thành composite descriptors
```python
# Nhóm các cặp liên quan:
overall_build  = mean(muscular_score, athletic_score, fit_score)       # 3 cặp → 1 score
height_related = mean(tall_score, long_legs_score, long_torso_score)   # 3 cặp → 1 score  
width_related  = mean(broad_shoulder_score, heavyset_score, stocky_score) # 3 cặp → 1 score
body_type      = mean(curvy_score, pear_score, full_figured_score)     # 3 cặp → 1 score
```

---

## Datasets nên đánh giá

| Dataset | Loại | Phù hợp | Ghi chú |
|---|---|---|---|
| Market-1501 | Short-term holistic | ✅ Baseline | So sánh trực tiếp với CLIP-ReID |
| DukeMTMC | Short-term holistic | ✅ | Cross-dataset evaluation |
| MSMT17 | Short-term, khó | ✅ | Chứng minh scalability |
| **PRCC** | **Clothes-changing** | ✅✅ | **Chứng minh body shape works** |
| **LTCC** | **Long-term CC** | ✅✅ | **Key dataset cho story** |
| Occ-Duke | Occluded | ✅ | Body shape robust với occlusion |

---

## Kết luận

**16 cặp từ body shape + CLIP zero-shot = source of body shape supervision hoàn toàn miễn phí.** Đây là lợi thế rất lớn mà không paper nào nói — ta không cần annotation, CLIP đã biết sẵn "muscular person" trông thế nào.

Lộ trình: **PP2 (chắc chắn, dễ) → PP4 (contribution mạnh) → evaluation trên LTCC/PRCC (story compelling)**.

Đặt tên phương pháp gợi ý: **BodyCLIP-ReID** hoặc **ShapePrompt-ReID**.
