import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Low-rank residual around a frozen nn.Linear.

    y = base(x) + (alpha/r) * B @ A @ x, with B initialised to zero so the
    wrapped forward matches the pretrained behaviour at step 0.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear wraps nn.Linear, got {type(base).__name__}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        out = self.base(x)
        lora_x = self.lora_dropout(x)
        lora_out = F.linear(F.linear(lora_x, self.lora_A), self.lora_B) * self.scaling
        return out + lora_out.to(out.dtype)


_LORA_TARGET_MAP = {
    "qkv": ("attn", "qkv"),
    "proj": ("attn", "proj"),
    "fc1": ("mlp", "fc1"),
    "fc2": ("mlp", "fc2"),
}


def _parse_block_indices(blocks_spec, num_blocks):
    if blocks_spec is None:
        return list(range(num_blocks))
    spec = str(blocks_spec).strip().lower()
    if spec in ("all", "*", ""):
        return list(range(num_blocks))
    indices = [int(b) for b in spec.split("_") if b != ""]
    for idx in indices:
        if idx < 0 or idx >= num_blocks:
            raise ValueError(f"LoRA block index {idx} out of range [0, {num_blocks})")
    return indices


def _parse_targets(targets_spec):
    if targets_spec is None:
        return ["qkv"]
    spec = str(targets_spec).strip()
    parts = [t.strip() for t in spec.replace("_", ",").split(",") if t.strip()]
    for t in parts:
        if t not in _LORA_TARGET_MAP:
            raise ValueError(
                f"Unknown LoRA target '{t}'; supported: {list(_LORA_TARGET_MAP)}"
            )
    return parts or ["qkv"]


def apply_dino_lora(
    dinov2_model,
    rank: int = 8,
    alpha: float = 16.0,
    targets="qkv",
    dropout: float = 0.0,
    blocks="all",
):
    """Freeze the DINOv2 ViT and inject LoRALinear into target Linears of selected blocks.

    Returns (num_injected, num_lora_params).
    """
    for p in dinov2_model.parameters():
        p.requires_grad_(False)

    num_blocks = len(dinov2_model.blocks)
    block_indices = _parse_block_indices(blocks, num_blocks)
    target_names = _parse_targets(targets)

    injected = 0
    lora_params = 0
    for i in block_indices:
        block = dinov2_model.blocks[i]
        for target in target_names:
            parent_attr, linear_attr = _LORA_TARGET_MAP[target]
            parent = getattr(block, parent_attr)
            base_linear = getattr(parent, linear_attr)
            if isinstance(base_linear, LoRALinear):
                continue
            wrapped = LoRALinear(base_linear, r=rank, alpha=alpha, dropout=dropout)
            setattr(parent, linear_attr, wrapped)
            injected += 1
            lora_params += wrapped.lora_A.numel() + wrapped.lora_B.numel()
    return injected, lora_params


def iter_lora_parameters(module):
    """Yield (name, param) pairs for LoRA parameters under a module."""
    for name, p in module.named_parameters():
        if name.endswith(".lora_A") or name.endswith(".lora_B") or ".lora_A" in name or ".lora_B" in name:
            yield name, p


def is_lora_param_name(name: str) -> bool:
    return "lora_A" in name or "lora_B" in name
