import torch
import torch.nn as nn


class MLPAdapter(nn.Module):
    """Bottleneck MLP adapter (SelaVPR-style parallel-to-MLP variant).

    Forward: y = up(GELU(down(x))). The up projection is zero-initialised so
    the adapter output is 0 at training start, preserving frozen-backbone
    forward.
    """

    def __init__(self, dim: int, ratio: float = 0.25, dropout: float = 0.0):
        super().__init__()
        if not (0 < ratio <= 1.0):
            raise ValueError(f"MLPAdapter ratio must be in (0, 1], got {ratio}")
        hidden = max(1, int(round(dim * ratio)))
        self.dim = dim
        self.hidden = hidden
        self.down = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.dropout(self.act(self.down(x))))


class _PTv3BlockWithMLPAdapter(nn.Module):
    """Wraps a PTv3 Block to inject MLPAdapter parallel to its MLP.

    Replicates the original Block.forward (Point-aware) and adds the adapter
    output to the MLP branch before the second residual addition. drop_path
    is applied to the combined (MLP + adapter) branch so stochastic depth
    drops both paths uniformly.
    """

    def __init__(self, base_block, adapter):
        super().__init__()
        self.block = base_block
        self.adapter = adapter

    def forward(self, point):
        block = self.block

        # First half: CPE + Attention (unchanged from original)
        shortcut = point.feat
        point = block.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if block.pre_norm:
            point = block.norm1(point)
        point = block.drop_path(block.ls1(block.attn(point)))
        point.feat = shortcut + point.feat
        if not block.pre_norm:
            point = block.norm1(point)

        # Second half: norm2 + (MLP + adapter parallel) + drop_path + residual
        shortcut = point.feat
        if block.pre_norm:
            point = block.norm2(point)
        normed_feat = point.feat
        adapter_out = self.adapter(normed_feat)

        point = block.ls2(block.mlp(point))
        point.feat = point.feat + adapter_out
        point = block.drop_path(point)
        point.feat = shortcut + point.feat
        if not block.pre_norm:
            point = block.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


def _parse_stage_indices(stages_spec, num_stages):
    if stages_spec is None:
        return list(range(num_stages))
    spec = str(stages_spec).strip().lower()
    if spec in ("all", "*", ""):
        return list(range(num_stages))
    indices = [int(s) for s in spec.split("_") if s != ""]
    for idx in indices:
        if idx < 0 or idx >= num_stages:
            raise ValueError(f"Stage index {idx} out of range [0, {num_stages})")
    return indices


def apply_utonia_mlp_adapter(
    ptv3_model,
    stages="4",
    ratio=0.25,
    dropout=0.0,
):
    """Freeze base PTv3 and wrap selected encoder Blocks with MLPAdapter.

    Returns (num_wrapped, num_adapter_params).
    """
    for p in ptv3_model.parameters():
        p.requires_grad_(False)

    enc = getattr(ptv3_model, "enc", None)
    if enc is None:
        raise RuntimeError("PTv3 model has no .enc attribute")

    num_stages = len(enc._modules)
    stage_indices = _parse_stage_indices(stages, num_stages)

    wrapped = 0
    adapter_params = 0
    for s in stage_indices:
        stage_name = f"enc{s}"
        stage = getattr(enc, stage_name, None)
        if stage is None:
            continue
        for name in list(stage._modules.keys()):
            mod = stage._modules[name]
            if not name.startswith("block"):
                continue
            if isinstance(mod, _PTv3BlockWithMLPAdapter):
                continue
            mlp_seq = getattr(mod, "mlp", None)
            if mlp_seq is None:
                continue
            inner = mlp_seq[0] if hasattr(mlp_seq, "__getitem__") else None
            if inner is None or not hasattr(inner, "fc1"):
                continue
            dim = inner.fc1.in_features
            adapter = MLPAdapter(dim=dim, ratio=ratio, dropout=dropout)
            stage._modules[name] = _PTv3BlockWithMLPAdapter(mod, adapter)
            wrapped += 1
            for p in adapter.parameters():
                adapter_params += p.numel()
    return wrapped, adapter_params
