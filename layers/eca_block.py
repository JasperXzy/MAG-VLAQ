# Implementation of Efficient Channel Attention ECA block
# Pure PyTorch version (no MinkowskiEngine dependency)

import numpy as np
import torch
import torch.nn as nn

from layers.sparse_utils import (
    SimpleSparse,
    sparse_global_avg_pool,
    sparse_broadcast_mul,
)


class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((np.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: SimpleSparse
        # Global average pool -> [B, C]
        y = sparse_global_avg_pool(x)

        # Apply 1D convolution along the channel dimension
        y = self.conv(y.unsqueeze(-1).transpose(-1, -2)).transpose(-1, -2).squeeze(-1)
        # y is (batch_size, channels) tensor

        y = self.sigmoid(y)

        # Broadcast multiply back to sparse features
        output = sparse_broadcast_mul(x, y)
        return output


class ECABasicBlock(nn.Module):
    """Pointwise sparse feature refinement block with ECA attention.
    Replaces the ME-based version that inherited from MinkowskiEngine.BasicBlock.
    Uses nn.Linear (pointwise transform) instead of spatial sparse convolutions,
    since this block is used for per-point feature refinement (same in/out dims).
    """
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, dimension=3):
        super().__init__()
        self.conv1 = nn.Linear(inplanes, planes)
        self.norm1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Linear(planes, planes)
        self.norm2 = nn.BatchNorm1d(planes)
        self.eca = ECALayer(planes, gamma=2, b=1)

        self.downsample = None
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Linear(inplanes, planes),
                nn.BatchNorm1d(planes),
            )

    def forward(self, x):
        # x: SimpleSparse
        residual = x

        out = SimpleSparse(features=self.conv1(x.F), coordinates=x.C)
        out = SimpleSparse(features=self.norm1(out.F), coordinates=out.C)
        out = SimpleSparse(features=self.relu(out.F), coordinates=out.C)

        out = SimpleSparse(features=self.conv2(out.F), coordinates=out.C)
        out = SimpleSparse(features=self.norm2(out.F), coordinates=out.C)
        out = self.eca(out)

        if self.downsample is not None:
            residual = SimpleSparse(
                features=self.downsample(residual.F),
                coordinates=residual.C,
            )

        out = SimpleSparse(features=out.F + residual.F, coordinates=out.C)
        out = SimpleSparse(features=self.relu(out.F), coordinates=out.C)
        return out
