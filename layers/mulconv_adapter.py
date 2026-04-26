import math

import torch
import torch.nn as nn


class MultiConvAdapter(nn.Module):
    """Multi-scale convolutional bottleneck adapter for ViT blocks (CricaVPR-style).

    Reshapes patch tokens [B, N, C] to a 2D grid [B, C, H, W], applies a 1x1
    bottleneck, then a parallel multi-kernel conv (e.g. 1x1 / 3x3 / 5x5), then
    a 1x1 up projection back to C. The up projection is zero-initialised so
    the adapter output starts at 0, preserving the frozen-backbone forward.
    """

    def __init__(self, dim: int, ratio: float = 0.5, kernels=(1, 3, 5)):
        super().__init__()
        if not (0 < ratio <= 1.0):
            raise ValueError(f"MultiConv ratio must be in (0, 1], got {ratio}")
        if not kernels:
            raise ValueError("MultiConv kernels list cannot be empty")
        for k in kernels:
            if k % 2 == 0:
                raise ValueError(f"MultiConv kernel must be odd, got {k}")

        hidden = max(len(kernels), int(round(dim * ratio)))
        per = [hidden // len(kernels)] * len(kernels)
        per[-1] += hidden - sum(per)

        self.dim = dim
        self.hidden = hidden
        self.kernels = list(kernels)

        self.down = nn.Conv2d(dim, hidden, kernel_size=1)
        self.convs = nn.ModuleList([
            nn.Conv2d(hidden, p, kernel_size=k, padding=k // 2)
            for k, p in zip(kernels, per)
        ])
        self.act = nn.GELU()
        self.up = nn.Conv2d(hidden, dim, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens):
        if tokens.dim() != 3:
            raise ValueError(f"MultiConv expects [B, N, C], got {tokens.shape}")
        B, N, C = tokens.shape
        if C != self.dim:
            raise ValueError(f"MultiConv expects channel {self.dim}, got {C}")
        side = int(math.isqrt(N))
        if side * side != N:
            raise ValueError(
                f"MultiConv expects square patch grid; got N={N}. "
                "Did you forget to drop the CLS token before calling?"
            )
        x = tokens.transpose(1, 2).reshape(B, C, side, side).contiguous()
        x = self.down(x)
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        x = self.act(x)
        x = self.up(x)
        return x.flatten(2).transpose(1, 2)


class _DinoBlockWithMultiConv(nn.Module):
    """Wraps a DINOv2 Block to inject MultiConvAdapter parallel to its MLP.

    Replicates the standard Block.forward (without stochastic depth path) and
    injects the adapter on the same input that MLP receives (norm2(x)). The
    CLS token is excluded from MultiConv because it is not on the patch grid.
    """

    def __init__(self, base_block, adapter):
        super().__init__()
        self.block = base_block
        self.adapter = adapter

    @staticmethod
    def _drop_path(block):
        return (
            getattr(block, "drop_path2", None)
            or getattr(block, "drop_path1", None)
            or getattr(block, "drop_path", None)
            or nn.Identity()
        )

    def forward(self, x):
        block = self.block
        # First residual: attention (unchanged)
        attn_branch = block.ls1(block.attn(block.norm1(x)))
        attn_drop = (
            getattr(block, "drop_path1", None)
            or getattr(block, "drop_path", None)
            or nn.Identity()
        )
        x = x + attn_drop(attn_branch)

        # Second residual: MLP + MultiConv adapter (parallel)
        normed = block.norm2(x)
        mlp_branch = block.ls2(block.mlp(normed))
        # MultiConv operates on patch tokens only (CLS at position 0)
        cls_tok = normed[:, :1, :]
        patch_tok = normed[:, 1:, :]
        adapter_patch = self.adapter(patch_tok)
        adapter_branch = torch.cat([torch.zeros_like(cls_tok), adapter_patch], dim=1)

        combined = mlp_branch + adapter_branch
        x = x + self._drop_path(block)(combined)
        return x


def _parse_block_indices(blocks_spec, num_blocks):
    if blocks_spec is None:
        return list(range(num_blocks))
    spec = str(blocks_spec).strip().lower()
    if spec in ("all", "*", ""):
        return list(range(num_blocks))
    indices = [int(b) for b in spec.split("_") if b != ""]
    for idx in indices:
        if idx < 0 or idx >= num_blocks:
            raise ValueError(f"MultiConv block index {idx} out of range [0, {num_blocks})")
    return indices


def _parse_kernels(kernels_spec):
    if kernels_spec is None:
        return (1, 3, 5)
    if isinstance(kernels_spec, (list, tuple)):
        return tuple(int(k) for k in kernels_spec)
    return tuple(int(k) for k in str(kernels_spec).split("_") if k)


def apply_dino_multiconv(
    dinov2_model,
    blocks="22_23",
    ratio=0.5,
    kernels=(1, 3, 5),
):
    """Freeze DINOv2 and wrap selected blocks with MultiConvAdapter.

    Returns (num_wrapped, num_adapter_params).
    """
    for p in dinov2_model.parameters():
        p.requires_grad_(False)

    if not hasattr(dinov2_model, "blocks"):
        raise RuntimeError("DINOv2 model has no .blocks attribute")

    num_blocks = len(dinov2_model.blocks)
    block_indices = _parse_block_indices(blocks, num_blocks)
    kernel_tuple = _parse_kernels(kernels)

    # Infer dim from first block's norm1
    first_block = dinov2_model.blocks[0]
    if hasattr(first_block, "norm1") and hasattr(first_block.norm1, "normalized_shape"):
        dim = first_block.norm1.normalized_shape[0]
    else:
        raise RuntimeError("Could not infer DINOv2 embed_dim from block norm1")

    wrapped = 0
    adapter_params = 0
    for i in block_indices:
        if isinstance(dinov2_model.blocks[i], _DinoBlockWithMultiConv):
            continue
        adapter = MultiConvAdapter(dim=dim, ratio=ratio, kernels=kernel_tuple)
        wrapped_block = _DinoBlockWithMultiConv(dinov2_model.blocks[i], adapter)
        dinov2_model.blocks[i] = wrapped_block
        wrapped += 1
        for p in adapter.parameters():
            adapter_params += p.numel()
    return wrapped, adapter_params
