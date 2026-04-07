# Pooling methods code based on: https://github.com/filipradenovic/cnnimageretrieval-pytorch
# Ported from MinkowskiEngine to pure PyTorch + torch_scatter

import torch
import torch.nn as nn
from layers.sparse_utils import SimpleSparse, sparse_global_avg_pool, sparse_global_max_pool


class PoolingWrapper(nn.Module):
    def __init__(self, pool_method, in_dim, output_dim):
        super().__init__()

        self.pool_method = pool_method
        self.in_dim = in_dim
        self.output_dim = output_dim

        if pool_method == 'MAC':
            assert in_dim == output_dim
            self.pooling = MAC(input_dim=in_dim)
        elif pool_method == 'SPoC':
            assert in_dim == output_dim
            self.pooling = SPoC(input_dim=in_dim)
        elif pool_method == 'GeM':
            assert in_dim == output_dim
            self.pooling = MinkGeM(input_dim=in_dim)
        else:
            raise NotImplementedError('Unknown pooling method: {}'.format(pool_method))

    def forward(self, x: SimpleSparse):
        return self.pooling(x)


class MAC(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = self.input_dim

    def forward(self, x: SimpleSparse):
        return sparse_global_max_pool(x)


class SPoC(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = self.input_dim

    def forward(self, x: SimpleSparse):
        return sparse_global_avg_pool(x)


class MinkGeM(nn.Module):
    def __init__(self, input_dim=None, p=3, eps=1e-6):
        super(MinkGeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: SimpleSparse):
        assert isinstance(x, SimpleSparse)
        # Clamp + power
        powered = SimpleSparse(
            features=x.F.clamp(min=self.eps).pow(self.p),
            coordinates=x.C,
        )
        # Global average pool
        temp = sparse_global_avg_pool(powered)
        output = temp.pow(1. / self.p)
        return output
