## ODE

### 一、整体架构流程

```
Query 输入
├── 2D 图像 ──→ DINOv2 (ImageFE) ──→ imagefeatmap + imagefeatmaplist (多层特征)
├── 3D 点云 ──→ Utonia/PTv3 (UtoniaFE) ──→ voxfeatmap + voxfeatmaplist (多层特征)
│
├── [独立分支] image_pool → image_proj → imagefeatvec_org   (2D 全局描述子)
├── [独立分支] vox_pool → vox_proj → voxfeatvec_org        (3D 全局描述子)
│
├── [Stage-1 融合] FuseBlockToShallow ──→ shallowfeatvec   (ODE)
├── [Stage-2 融合] Stage2FuseBlockAdd ──→ stg2fusevec
│
└── final output = 加权求和(所有分支)
```

ODE 用于 **Stage-1 多模态特征融合**，将 DINOv2 提取的 2D 图像特征和 Utonia 提取的 3D 点云特征，通过 Neural ODE 动态系统逐层融合为一个统一的表征向量

### 二、ODE 具体代码链路

#### 2.1 `fuse_block_toshallow.py` → `forward_imgvox()`

将多层 2D/3D 特征从 deep → shallow 逐层融合：

```python
fusevec = 0
for i in range(len(self.dims)):
    i = len(self.dims)-1-i                    # backward: 从深层到浅层
    imagevec = updimsimg[i](imageveclist[i])  # 2D 特征投影到统一维度
    voxvec   = updimsvox[i](voxveclist[i])    # 3D 特征投影到统一维度

    fusevec = fusevec + imagevec + voxvec     # 残差累加
    fusevec = block(fusevec)                  # DiffBlock，内含 ODE
```

每一层都：先把 2D 和 3D 特征相加到 `fusevec`，然后通过 `DiffBlock` 做一次 ODE 演化

#### 2.2 `diff_block.py` → `DiffBlock`

```python
class DiffBlock(nn.Module):
    def __init__(self, dim, ode_dim):
        # 默认 "fcode@relu"
        for e in diff_type.split('_'):
            e, act = e.split('@')                    # e="fcode", act="relu"
            if e == 'fcode':
                self.blocks.append(FCODE(dim, act))  # Neural ODE 模块

    def forward(self, x):
        for block in self.blocks:
            out = block(x)
        return sum(outlist)
```

#### 2.3 `ffns.py` → `FCODE`（核心 ODE 求解）

```python
class ODEFunc(nn.Module):
    """定义 ODE 的导数函数 dx/dt = f(x)"""
    def __init__(self, func):
        self.func = func           # FC(dim, dim, relu)
    def forward(self, t, x):
        return self.func(x)        # dx/dt = ReLU(Linear(x))

class FCODE(nn.Module):
    """用 odeint 求解 ODE 初值问题"""
    def __init__(self, dim, act='relu'):
        self.func = ODEFunc(FC(dim, dim, act))

    def forward(self, x):
        t = torch.tensor([0, 1]).float()  # 从 t=0 积分到 t=1
        out = odeint(
            self.func, x, t,
            method='euler',               # 默认 Euler 法
            options={'step_size': 0.1},   # 步长 0.1 → 10 步
            rtol=1e-3, atol=1e-3
        )
        return out[-1]  # 取 t=1 时刻的状态
```

给定初值 `x(0) = fusevec`，求解：

$$\frac{dx}{dt} = \text{ReLU}(\text{Linear}(x)), \quad t \in [0, 1]$$

用 Euler 法步长 0.1，实际展开为 10 步迭代：

```
x_{k+1} = x_k + 0.1 * ReLU(W @ x_k + b)
```

### 三、Neural ODE 的优势分析

#### 3.1 参数效率高

```
普通做法：10 层 MLP → 10 组 (W, b)，参数量 = 10 × dim²
ODE：1 组 (W, b)，Euler 步长 0.1 等效 10 步迭代，参数量 = 1 × dim²
```

默认 `dim=256`，ODE 只用 **65K 参数** 就实现了等效 10 层网络，在多模态融合场景下有多个 `DiffBlock`（每个 encoder stage 一个），参数节省非常显著，降低过拟合风险

#### 3.2 连续动力系统 → 特征平滑演化

普通 MLP 是离散映射 `x → f(x)`，ODE 把特征变换建模为连续流：

$$x(t+\Delta t) = x(t) + \Delta t \cdot \text{ReLU}(Wx(t) + b)$$

- 2D 图像特征和 3D 点云特征处于不同的表征空间，直接相加后存在模态间隙
- ODE 的连续演化提供一条 **平滑的融合路径**，让两种模态的特征逐步对齐，而非强行一步映射

#### 3.3 梯度稳定性

普通深层网络的反向传播需要存储所有中间激活值，且容易出现梯度爆炸/消失，但是 odeint 权重共享使梯度天然有平均效果，不易爆炸

#### 3.5 与逐层融合策略契合

Stage-1 融合是 **deep-to-shallow 逐层累加**：

```python
fusevec = 0
for i in [深层 → 浅层]:
    fusevec = fusevec + imagevec[i] + voxvec[i]  # 每层注入新信息
    fusevec = ODE(fusevec)                       # ODE 演化消化新信息
```

每次注入一层新的 2D+3D 特征后，ODE 做一次连续演化

- **concat + MLP**：一次性处理所有层，丢失层级关系
- **逐层 MLP**：每层独立权重，参数多且层间无关联
- **逐层 ODE**：共享动力学，每一层的演化方式一致，符合"同一个融合规则应用于不同尺度"的直觉

#### 3.6 对比其他融合方法

|方法|参数量|深度可调|内存效率|融合质量|
|---|---|---|---|---|
|Concat + FC|dim²×N|不可调|低|粗暴，易过拟合|
|Cross-Attention|3×dim²×N|不可调|低（N² 注意力矩阵）|高但昂贵|
|逐层 ResBlock|dim²×N 层|需改架构|中|好|
|**Neural ODE**|**dim²×1**|**一个超参**|**高（adjoint）**|**好，且平滑**|
|GNN (Beltrami)|dim²×1|一个超参|中|更强但更慢|

### 四、劣势

- **推理速度**：Euler 10 步 = 10 次前向，比单层 MLP 慢约 10 倍
- **表达能力上限**：权重共享限制了每一步可以做不同变换的能力，极端情况下不如独立参数的深层网络
- **ODE 求解器选择敏感**：Euler 法精度低但快，高阶方法精度高但慢，需要调参
