import torch.nn as nn

from layers.sparse_utils import SimpleSparse


class Chart2D(nn.Module):
    def __init__(self, C_in, C_out):
        super().__init__()
        self.proj = nn.Linear(C_in, C_out)

    def forward(self, x):
        return self.proj(x)


class Chart3D(nn.Module):
    def __init__(self, C_in, C_out):
        super().__init__()
        self.proj = nn.Linear(C_in, C_out)

    def forward(self, x):
        return SimpleSparse(features=self.proj(x.F), coordinates=x.C)
