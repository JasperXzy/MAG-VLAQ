# SCA-AGPlace Backbone 架构详解

### 1. Overall Data Flow

```
INPUT
├── query_image: [B, 3, 256, 256]          <- 2D RGB image
├── utonia_coord: [N_total, 3]             <- raw float XYZ coordinates
├── utonia_grid_coord: [N_total, 3]        <- quantized int grid coordinates
├── utonia_feat: [N_total, 3]              <- XYZ features (overwritten to 9ch in forward)
├── utonia_rgb: [N_total, 3]               <- per-point RGB from fisheye camera projection (0-1)
└── utonia_offset: [B]                     <- cumulative point counts per batch


  ┌────────────────────────────┐          ┌──────────────────────────────┐
  │  ImageFE (DINOv2)          │          │  UtoniaFE (PTv3)             │
  │  network_mm/image_fe.py    │          │  network_mm/utonia_fe.py     │
  │                            │          │                              │
  │  Output:                   │          │  Output:                     │
  │  imagefeatmap [B,C,H,W]    │          │  voxfeatmap (SimpleSparse)   │
  │  imagefeatmaplist x3       │          │  voxfeatmaplist x3           │
  └───┬────────────┬───────────┘          └───┬────────────┬─────────────┘
      │            │                          │            │
      │(featmap)   │(maplist x3)              │(featmap)   │(maplist x3)
      │            │                          │            │
      ▼            │                          ▼            │
   GeM Pool        │                       MinkGeM         │
   flatten         │                       flatten         │
   MLP proj        │                       MLP proj        │
      │            │                          │            │
 imagevec_org      │                     voxvec_org        │
   [B, 256]    ────│──────────────────────  [B, 256]       │
      │            │                          │            │
      │            └──────────┬───────────────┘            │
      │                       │                            │
      │            ┌──────────▼────────────────────┐       │
      │            │  Stage-1 Fusion (Early)       │<──────┘
      │            │  FuseBlockToShallow           │
      │            │  fuse_block_toshallow.py      │
      │            │                               │
      │            │  Input: imagemaplist x3       │
      │            │       + voxmaplist x3         │
      │            │  Pool each layer -> align dim │
      │            │  -> residual accumulation     │
      │            │  -> DiffBlock (Neural ODE)    │
      │            └────────────┬──────────────────┘
      │                         │
      │                  shallowfeatvec [B, 256]
      │                         │
      │                         │ x shallow_weight
      │                         ▼
      │  ┌──────────────────────────────────────────────────┐
      │  │  Stage-2 Fusion (Late)                           │
      │  │  Stage2FuseBlockAdd                              │
      │  │  stage2fuse_blockadd.py                          │
      │  │                                                  │
      │  │  Input: imagefeatmap + voxfeatmap                │
      │  │       + shallowvec x w (as initial fusevec)      │
      │  │  Broadcast add -> BasicBlock / ECABasicBlock     │
      │  │  -> Pool -> back-project -> FFN refine           │
      │  └───┬──────────────────┬───────────────────────────┘
      │      │                  │
      │  stg2imagevec       stg2voxvec
      │   [B, 256]           [B, 256]
      │      │                  │
      ▼      ▼                  ▼
  ┌──────────────────────────────────────────────────┐
  │  Final Weighted Combination                      │
  │                                                  │
  │  embedding = imagevec_org   x w_imgorg           │
  │            + voxvec_org     x w_voxorg           │
  │            + shallowfeatvec x w_shallow          │
  │            + stg2imagevec   x w_stg2img          │
  │            + stg2voxvec     x w_stg2vox          │
  │                                                  │
  │  -> L2 normalize                                 │
  └──────────────────────┬───────────────────────────┘
                         │
                    embedding [B, 256]
                    (final scene retrieval descriptor)
```


## 二、2D 分支：ImageFE (DINOv2)

### 2.1 模型加载

```python
self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
# ViT-Large, patch_size=14, embed_dim=1024, 24 transformer blocks
```

### 2.2 冻结策略

| 模式 (`unfreeze_dino_mode`) | 行为 |
|---|---|
| `'frozen'` (默认) | 全部冻结，纯特征提取器 |
| `'last2'` | 解冻最后 2 个 transformer block + 最终 LayerNorm |
| `'full'` | 全部可训练 |

### 2.3 多层特征提取

```python
# 默认提取 block 7, 15, 23 (dino_extract_blocks="7_15_23")
# 对应 DINOv2 的浅层、中层、深层

for i, blk in enumerate(self.backbone.blocks):
    tokens = blk(tokens)
    if i in target_blocks:
        feat = tokens[:, 1:]              # 去掉 CLS token
        feat = feat.permute(0,2,1)        # [B, 1024, num_patches]
        feat = feat.reshape(B, 1024, H//14, W//14)  # 恢复空间结构
        x_list.append(feat)
```

### 2.4 输出

```python
feat_map:  Tensor [B, 1024, H/14, W/14]    # 最后一层特征图
x_list:    [Tensor × 3]                    # 3 层特征图 (用于 Stage-1 融合)
```

**ATTN**: 使用 DINOv2 时，所有 3 层的维度都是 1024（不像 ResNet 有 64→128→256 的递增）， `img_dims` 被统一设为 `[1024, 1024, 1024]`


## 三、3D 分支：UtoniaFE

### 3.1 预训练加载与适配

```python
# 从本地缓存或 HuggingFace 加载 PTv3 预训练权重
# --utonia_pretrained ~/.cache/utonia/ckpt/utonia.pth (本地路径，离线)
# --utonia_pretrained utonia                          (HF 下载)
ckpt = utonia_load(pretrained_name, ckpt_only=True)
config = ckpt['config']

# 配置
config['enc_mode'] = True                              # 仅编码器，无解码器
config['traceable'] = True                             # 记录 pooling_parent 链
config['enable_flash'] = True                          # Flash Attention
config['enc_patch_size'] = [256, 256, 512, 512, 1024]  # 渐进式 patch 大小
config['drop_path'] = 0.1                              # 随机深度正则化
```

### 3.2 输入通道：9ch (xyz + rgb + normal)

保持与 Utonia 预训练一致的 9 通道接口，预训练权重直接加载，无需适配：

```python
# 预训练模型: in_channels=9 (xyz + rgb + normal)
# 本项目输入: 9ch — xyz(坐标归一化) + rgb(鱼眼相机投影) + normal(PCA 估计)
# Embedding 层 shape 完全匹配，load_state_dict 直接命中
#
# RGB 来源: KITTI-360 鱼眼相机 image_02 (datasets/kitti360_calib.py)
#   - 通过标定参数将 3D 点投影到鱼眼图像采样像素颜色
#   - 不在相机 FOV 内的点 RGB = [0,0,0]
#   - 符合 Utonia 论文的 Causal Modality Blinding 设计
```

### 3.3 PTv3 编码器架构

#### 总体结构：5 个 Encoder Stage

| Stage | 通道数 | 注意力头数 | Patch 大小 | 下采样步长 | Block 数 | 特点 |
|-------|--------|-----------|-----------|-----------|---------|------|
| 0 | 48 | 2 | 256 | - | 2 | 初始局部特征，无下采样 |
| 1 | 96 | 4 | 256 | 2× | 2 | 2× 下采样 |
| 2 | 192 | 8 | 512 | 2× | 2 | 4× 下采样 |
| 3 | 384 | 16 | 512 | 2× | 6 | 8× 下采样，最深（6 blocks） |
| 4 | 512 | 16 | 1024 | 2× | 2 | 16× 下采样，最大感受野 |

#### Embedding 层

```python
class Embedding(PointModule):
    # 输入: 9ch (xyz + rgb + normal)
    # 输出: 48ch (from-scratch) 或预训练配置的 embed_ch
    stem = Linear(9, embed_ch) + LayerNorm(embed_ch) + GELU
```

#### 单个 Encoder Block 内部结构

```
输入 Point (feat: [N, C])
    │
    ├── 1. CPE (Contextual Point Embedding)
    │       SubMConv3d(C → C, kernel=3) + Linear(C → C) + LayerNorm
    │       作用: 局部空间上下文编码
    │
    ├── 2. Serialized Attention
    │       ├── 多种序列化顺序: z-order, z-trans, hilbert, hilbert-trans
    │       ├── 点按空间填充曲线排序
    │       ├── 分组为 patch (大小 = enc_patch_size)
    │       ├── 每个 patch 内做多头自注意力 + RoPE 位置编码
    │       └── 多种序列化结果取平均 (旋转不变性)
    │
    ├── 3. MLP (Feed-Forward)
    │       Linear(C → 4C) + GELU + Linear(4C → C)
    │       expansion ratio = 4
    │
    ├── 4. Drop Path (随机深度)
    │       训练时以概率 p 跳过当前 block
    │
    └── 5. 残差连接
            output = input + drop_path(block(input))
```

#### GridPooling (Stage 间下采样)

```python
class GridPooling(PointModule):
    # 将点聚合到更粗的网格
    # 1. grid_coord = coord // stride  (整数除法量化)
    # 2. 同一网格内的点: 特征取 max/mean, 坐标取 mean
    # 3. 记录 pooling_parent, 用于多层特征回溯
```

### 3.4 Forward 流程

```
输入 data_dict
    │
    ├── 1. 坐标归一化 (per-batch)
    │       grid_coord -= min_per_batch      # 平移到原点
    │       coord -= min_per_batch
    │       coord /= max_range               # 缩放到单位范围
    │       目的: 室外大尺度坐标 → 匹配预训练室内数据分布
    │
    ├── 2. 法线估计
    │       _estimate_normals(coord, offset, k=20)
    │       ├── 分 chunk 计算 KNN (避免 O(N²) 内存)
    │       ├── 对每个点的 k 近邻计算协方差矩阵
    │       ├── 特征值分解，取最小特征值对应的特征向量 = 法线
    │       └── 返回 normals: [N, 3]
    │
    ├── 3. 构建 9ch 特征
    │       rgb = data_dict['rgb']   (鱼眼投影 RGB, FOV 外为零)
    │       feat = cat([normalized_coord, rgb, normals], dim=1)  → [N, 9]
    │
    ├── 4. PTv3 编码器前向传播
    │       Point → Embedding → Stage0 → Pool → Stage1 → ... → Stage4
    │       返回: Point 对象 (含 pooling_parent 链)
    │
    ├── 5. 多层特征提取 (★ 关键)
    │       通过 pooling_parent 链回溯:
    │       point → point.pooling_parent → ... (逐级向上)
    │       收集 all_stages = [stage4, stage3, stage2, stage1, stage0]
    │       反转为 [stage0, stage1, stage2, stage3, stage4]
    │       按 utonia_extract_stages="1_2_3" 选取 stage 1, 2, 3
    │
    └── 6. 投影为 SimpleSparse
            对每个选中的 stage:
            coords = [batch_idx, x, y, z]          # [Ni, 4]
            feats  = proj[i](stage_feat)            # [Ni, planes[i]]
            → SimpleSparse(features, coordinates)
```

### 3.5 特征提取位置与维度

默认 `utonia_extract_stages="1_2_3"`, `mm_voxfe_planes="64_128_256"`:

| 提取目标 | PTv3 原始通道 | 投影后通道 | 点数 (相对 Stage0) | 用途 |
|---------|-------------|----------|-------------------|------|
| Stage 1 | 96 | 64 | N/4 | 浅层局部特征 → Stage-1 融合 |
| Stage 2 | 192 | 128 | N/16 | 中层特征 → Stage-1 融合 |
| Stage 3 | 384 | 256 | N/64 | 深层语义特征 → Stage-1 融合 + 池化 |

```python
self.projs = nn.ModuleList([
    nn.Linear(96,  64),    # stage1 → 64ch
    nn.Linear(192, 128),   # stage2 → 128ch
    nn.Linear(384, 256),   # stage3 → 256ch
])
```

最终输出：
- `voxfeatmap`: 最后一个 stage 的 SimpleSparse (stage3, 256ch)
- `voxfeatmaplist`: 所有选中 stage 的 SimpleSparse 列表 (3 个)


## 四、Stage-1 融合：FuseBlockToShallow (ODE)

### 4.1 目的

将 2D (DINOv2) 和 3D (Utonia) 的多层特征逐层融合为单一向量

### 4.2 处理流程

```
输入:
  imagefeatmaplist: [Tensor × 3]      ← 3 层 DINOv2 特征图
  voxfeatmaplist:   [SimpleSparse × 3] ← 3 层 Utonia 稀疏特征

Step 1: 全局池化 → 向量
  imageveclist[i] = adaptive_avg_pool2d(img[i], 1).flatten(1)  → [B, 1024]
  voxveclist[i]   = sparse_global_avg_pool(vox[i])             → [B, planes[i]]

Step 2: 维度对齐
  imageveclist[i] = Linear(1024 → 256)(imageveclist[i])        → [B, 256]
  voxveclist[i]   = Linear(planes[i] → 256)(voxveclist[i])     → [B, 256]

Step 3: 逐层融合 (backward: 从深层到浅层)
  fusevec = 0
  for i in [2, 1, 0]:    # backward 方向
      fusevec = fusevec + imageveclist[i] + voxveclist[i]
      fusevec = DiffBlock(fusevec)    # Neural ODE 演化

输出: shallowfeatvec [B, 256]
```

### 4.3 DiffBlock → FCODE (ODE)

```
DiffBlock(fusevec)
    └── FCODE(fusevec)
            │
            ├── 定义 ODE: dx/dt = ReLU(Linear(x))
            ├── 初值: x(0) = fusevec
            ├── 求解: odeint(func, x, t=[0,1], method='euler', step=0.1)
            ├── 等效: 10 步迭代 x_{k+1} = x_k + 0.1 * ReLU(W·x_k + b)
            └── 返回: x(1)
```

---

## 五、Stage-2 融合：Stage2FuseBlockAdd

### 5.1 目的

以 Stage-1 产出的 `shallowfeatvec` 为引导，精炼原始 2D/3D 特征图，产出更强的模态表征

### 5.2 处理流程 (type='vox', stg2nlayers=1)

```
输入:
  imgmap:   [B, 1024, H/14, W/14]    ← DINOv2 最后一层特征图
  voxmap:   SimpleSparse [N, 256]     ← Utonia 最后一层稀疏特征
  fusevec:  [B, 256]                  ← Stage-1 的 shallowfeatvec

for each layer (default: 1 层):

  ┌── 1. 投影 fusevec → 各模态维度
  │     fusevec_img = Linear(256 → 1024)(fusevec)      → [B, 1024]
  │     fusevec_vox = Linear(256 → 256)(fusevec)       → [B, 256]
  │
  ├── 2. 广播加法: 将融合向量注入特征图
  │     imgmap = imgmap + fusevec_img[:,:,None,None]    ← 空间广播
  │     voxmap = sparse_broadcast_add(voxmap, fusevec_vox)
  │
  ├── 3. 特征精炼
  │     imgmap = BasicBlock(imgmap)                     ← Conv-BN-ReLU × 2 + 残差
  │     voxmap = ECABasicBlock(voxmap)                  ← Linear-BN-ReLU × 2 + ECA 通道注意力 + 残差
  │
  ├── 4. 池化为向量
  │     imgoutvec = GeM(imgmap).flatten(1)              → [B, 1024]
  │     voxoutvec = MinkGeM(voxmap).flatten(1)          → [B, 256]
  │
  └── 5. 反向投影 → 更新 fusevec
        imgmap_fuse = Conv2d(1024 → 256)(imgmap)
        voxmap_fuse = SparseLinear(256 → 256)(voxmap)
        imgvec_fuse = adaptive_avg_pool(imgmap_fuse)    → [B, 256]
        voxvec_fuse = sparse_global_avg_pool(voxmap_fuse) → [B, 256]
        fusevec = fusevec + imgvec_fuse + voxvec_fuse
        fusevec = Basic(fusevec)                        ← Linear-LN-ReLU × 2 + 残差

输出:
  stg2fusevec:  [B, 256]    ← 精炼后的融合向量
  stg2imagevec: [B, 1024]   ← 精炼后的图像向量
  stg2voxvec:   [B, 256]    ← 精炼后的点云向量
```

## 六、最终输出组合

### 6.1 各分支输出与投影

```
imagefeatvec_org  ← image_proj: MLP(1024 → 256)    → [B, 256]
voxfeatvec_org    ← vox_proj:   MLP(256  → 256)    → [B, 256]
shallowfeatvec    ← (已经是 256)                    → [B, 256]
stg2imagevec      ← stg2image_proj: MLP(1024→ 256) → [B, 256]
stg2voxvec        ← stg2vox_proj:   MLP(256 → 256) → [B, 256]
stg2fusevec       ← stg2fusefc: Linear(256 → 256)  → [B, 256]
```

### 6.2 加权求和

```python
# 默认 final_type = ['imageorg', 'voxorg', 'shalloworg', 'stg2image', 'stg2vox']
# 默认权重:
#   imageorg_weight = imagevoxorg_weight (默认值)
#   voxorg_weight   = imagevoxorg_weight
#   shallow_weight  = shalloworg_weight
#   stg2image_weight = stg2imagevox_weight = 0.1
#   stg2vox_weight   = stg2imagevox_weight = 0.1

finaloutput = [
    imagefeatvec_org  × imageorg_weight,
    voxfeatvec_org    × voxorg_weight,
    shallowfeatvec    × shalloworg_weight,
    stg2imagevec      × stg2image_weight,
    stg2voxvec        × stg2vox_weight,
]

# final_fusetype = 'add'
embedding = sum(finaloutput)       → [B, 256]
embedding = F.normalize(embedding) → [B, 256]  (L2 归一化)
```

### 6.3 输出字典

```python
output_dict = {
    'imagevec_org':   [B, 256],    # 独立 2D 描述子
    'voxvec_org':     [B, 256],    # 独立 3D 描述子
    'shallowvec_org': [B, 256],    # Stage-1 ODE 融合描述子
    'stg2fusevec':    [B, 256],    # Stage-2 融合描述子
    'stg2imagevec':   [B, 256],    # Stage-2 精炼 2D 描述子
    'stg2voxvec':     [B, 256],    # Stage-2 精炼 3D 描述子
    'embedding':      [B, 256],    # 最终场景检索向量
}
```

## 七、池化层

### 7.1 GeM (Generalized Mean Pooling)

```python
# 公式: (mean(x^p))^(1/p)
# p=1: 平均池化; p→∞: 最大池化; p=3(默认): 折中
class GeM(nn.Module):
    def __init__(self, p=3):
        self.p = nn.Parameter(torch.ones(1) * p)  # 可学习的 p
    def forward(self, x):  # [B, C, H, W]
        return avg_pool2d(x.clamp(min=eps).pow(self.p), (H, W)).pow(1./self.p)
        # 输出: [B, C, 1, 1]
```

### 7.2 MinkGeM (稀疏版 GeM)

```python
class MinkGeM(nn.Module):
    def forward(self, x: SimpleSparse):
        powered = SimpleSparse(x.F.clamp(min=eps).pow(self.p), x.C)
        avg = sparse_global_avg_pool(powered)   # scatter_add + count → [B, C]
        return avg.clamp(min=eps).pow(1./self.p)
        # 输出: [B, C]
```

### 7.3 ECA (Efficient Channel Attention)

```python
class ECALayer(nn.Module):
    # 自适应 1D 卷积核大小: k = f(channels)
    # channels=256 → k=7
    def forward(self, x: SimpleSparse):
        y = sparse_global_avg_pool(x)              # [B, C]
        y = self.conv(y.unsqueeze(-1).T).T          # 1D 卷积: 跨通道局部交互
        y = torch.sigmoid(y)                        # 通道注意力权重
        return sparse_broadcast_mul(x, y)           # 逐点加权
```

---

## 八、SimpleSparse 数据结构

MinkowskiEngine 的轻量替代品，纯 PyTorch 实现：

```python
class SimpleSparse:
    F: Tensor [N, C]       # 所有点的特征
    C: Tensor [N, 4]       # 坐标 [batch_idx, x, y, z]

    @property
    def batch_indices(self):
        return self.C[:, 0].long()

    @property
    def num_batches(self):
        return self.C[:, 0].max().item() + 1
```

核心操作（无外部依赖）：

| 操作 | 实现 | 用途 |
|------|------|------|
| `sparse_global_avg_pool(x)` | `scatter_add_(0, idx, F)` / counts | 稀疏→稠密 [B,C] |
| `sparse_global_max_pool(x)` | `scatter_reduce_('amax')` | 最大池化 |
| `sparse_broadcast_add(x, v)` | `F + v[batch_idx]` | 向量广播加 |
| `sparse_broadcast_mul(x, v)` | `F * v[batch_idx]` | 向量广播乘 |

---

## 九、数据预处理与 Collate

### 9.1 KITTI-360 数据集输出

每个样本包含：
- `query_image`: [3, 256, 256] — RGB 图像
- `query_pc`: [Ni, 3] — 原始点云 (float xyz)
- `query_pc_rgb`: [Ni, 3] — 逐点 RGB (鱼眼相机投影, 0-1; FOV 外为零)

### 9.2 Collate 函数

```python
def kitti360_collate_fn(batch):
    # 1. 图像堆叠
    query_image = torch.stack(images)                  # [B, 3, 256, 256]

    # 2. 点云拼接 + 旋转增强
    raw_coords = torch.cat(point_clouds)               # [N_total, 3]
    raw_coords_rotated = PCRandomRotation()(raw_coords) # 随机旋转

    # 3. ME 格式 (legacy)
    batch_ids = [torch.full((ni, 1), i) for i, ni in enumerate(counts)]
    coords = cat([batch_ids, raw_coords_rotated.int()])  # [N, 4]
    features = ones([N, 1])                               # 虚拟特征

    # 4. Utonia 格式
    utonia_offset = cumsum(tensor(point_counts))          # [B]
    utonia_coord = raw_coords_rotated.float()             # [N, 3] 原始精度
    utonia_grid_coord = raw_coords_rotated.int()          # [N, 3] 网格量化
    utonia_feat = raw_coords_rotated.float()              # [N, 3] 初始特征 (会被覆盖)
    utonia_rgb = cat(query_pc_rgb_list)                   # [N, 3] 鱼眼投影 RGB (0-1)
```

**设计要点**：
- 旋转在 **float 精度**下执行，保留子体素信息
- `utonia_feat` 初始为 xyz 坐标，在 UtoniaFE forward 中被覆盖为 `[xyz_norm, rgb, normals]` (9ch)
- `utonia_rgb` 在 `__getitem__` 中通过鱼眼相机标定投影获得，逐点关联不受旋转影响
- `utonia_offset` 为累积计数，用于 UtoniaFE 中的 per-batch 归一化和法线估计

## 十、完整前向传播数据流图

```
query_image [B,3,256,256]          utonia_coord [N,3]
       │                           utonia_grid_coord [N,3]
       │                           utonia_offset [B]
       │                                │
       ▼                                ▼
┌──────────────┐              ┌───────────────────────┐
│  DINOv2      │              │  UtoniaFE             │
│  ViT-L/14    │              │                       │
│              │              │  coord_normalize()    │
│  24 blocks   │              │  estimate_normals()   │
│  extract:    │              │  feat=[xyz,rgb,normal]│
│  block 7  ──────┐           │                       │
│  block 15 ──────┤           │  PTv3 Encoder:        │
│  block 23 ──────┤           │  Stage0 (48ch)        │
│              │  │           │  Stage1 (96ch)  ────────┐
│              │  │           │  Stage2 (192ch) ────────┤
│              │  │           │  Stage3 (384ch) ────────┤
│              │  │           │  Stage4 (512ch)      │  │
└──────┬───────┘  │           └──────────┬───────────┘  │
       │          │                      │              │
       │     img_list ×3           vox_last        vox_list ×3
       │     [B,1024,H,W]      SimpleSparse       SimpleSparse
       │          │             [N,256]            [64,128,256]
       │          │                  │                  │
       ▼          │                  ▼                  │
   GeM Pool       │             MinkGeM Pool            │
   [B,1024]       │             [B,256]                 │
       │          │                  │                  │
   MLP(1024→256)  │             MLP(256→256)            │
       │          │                  │                  │
  imagevec_org    │             voxvec_org              │
   [B,256]        │              [B,256]                │
       │          │                  │                  │
       │          │     ┌────────────┘                  │
       │          │     │                               │
       │          ▼     │              ▼                │
       │    ┌─────────────────────────────────────┐     │
       │    │     FuseBlockToShallow              │     │
       │    │                                     │     │
       │    │   for i in [deep → shallow]:        │◄────┘
       │    │     pool(img[i]) + pool(vox[i])     │
       │    │     → Linear 对齐到 256              │
       │    │     → fusevec += img + vox          │
       │    │     → fusevec = ODE(fusevec)        │
       │    │                                     │
       │    └──────────────┬──────────────────────┘
       │                   │
       │            shallowfeatvec [B,256]
       │                   │
       ▼                   ▼                ▲
  ┌────────────────────────────────────────────────┐
  │     Stage2FuseBlockAdd                         │
  │                                                │
  │  fusevec_img = proj(shallowvec)                │
  │  imgmap += broadcast(fusevec_img)              │
  │  imgmap = BasicBlock(imgmap)                   │
  │                                                │
  │  fusevec_vox = proj(shallowvec)                │
  │  voxmap += broadcast(fusevec_vox)              │
  │  voxmap = ECABasicBlock(voxmap)                │
  │                                                │
  │  → 池化 → 反向投影 → FFN 精炼                     │
  └───┬──────────────┬──────────────┬──────────────┘
      │              │              │
  stg2fusevec   stg2imagevec   stg2voxvec
   [B,256]       [B,256]       [B,256]
      │              │              │
      ▼              ▼              ▼
  ┌────────────────────────────────────────────────┐
  │  Final 加权求和                                 │
  │                                                │
  │  embedding = imagevec_org   × w_img            │
  │            + voxvec_org     × w_vox            │
  │            + shallowfeatvec × w_shallow        │
  │            + stg2imagevec   × w_stg2img        │
  │            + stg2voxvec     × w_stg2vox        │
  │                                                │
  │  embedding = L2_normalize(embedding)           │
  └────────────────────┬───────────────────────────┘
                       │
                  embedding [B, 256]
                  (场景检索最终描述子)
```
