"""
Lightweight replacement for MinkowskiEngine sparse tensor operations.
Pure PyTorch implementation (no torch_scatter dependency).
"""

import numpy as np
import torch


class SimpleSparse:
    """
    Minimal sparse tensor container, API-compatible with ME.SparseTensor
    for the subset of operations used in this project.

    Attributes:
        F: features [N, C]
        C: coordinates [N, 4] where column 0 is the batch index
    """

    def __init__(self, features, coordinates):
        self.F = features
        self.C = coordinates

    @property
    def batch_indices(self):
        return self.C[:, 0].long()

    @property
    def num_batches(self):
        return self.batch_indices.max().item() + 1


def sparse_global_avg_pool(x):
    """Global average pooling over sparse points, grouped by batch index.
    Args:
        x: SimpleSparse
    Returns:
        Tensor [B, C]
    """
    B = x.num_batches
    idx = x.batch_indices.unsqueeze(1).expand_as(x.F)
    sums = torch.zeros(B, x.F.shape[1], device=x.F.device, dtype=x.F.dtype)
    sums.scatter_add_(0, idx, x.F)
    counts = torch.zeros(B, device=x.F.device, dtype=x.F.dtype)
    counts.scatter_add_(0, x.batch_indices, torch.ones(x.F.shape[0], device=x.F.device, dtype=x.F.dtype))
    return sums / counts.unsqueeze(1).clamp(min=1)


def sparse_global_max_pool(x):
    """Global max pooling over sparse points, grouped by batch index.
    Args:
        x: SimpleSparse
    Returns:
        Tensor [B, C]
    """
    B = x.num_batches
    idx = x.batch_indices.unsqueeze(1).expand_as(x.F)
    out = torch.full((B, x.F.shape[1]), float('-inf'), device=x.F.device, dtype=x.F.dtype)
    out.scatter_reduce_(0, idx, x.F, reduce='amax')
    return out


def sparse_broadcast_add(sparse, vec):
    """Add a per-batch vector [B, C] to every point in the sparse tensor.
    Args:
        sparse: SimpleSparse with features [N, C]
        vec: Tensor [B, C]
    Returns:
        SimpleSparse with updated features
    """
    return SimpleSparse(
        features=sparse.F + vec[sparse.batch_indices],
        coordinates=sparse.C,
    )


def sparse_broadcast_mul(sparse, vec):
    """Multiply every point in the sparse tensor by a per-batch vector [B, C].
    Args:
        sparse: SimpleSparse with features [N, C]
        vec: Tensor [B, C]
    Returns:
        SimpleSparse with updated features
    """
    return SimpleSparse(
        features=sparse.F * vec[sparse.batch_indices],
        coordinates=sparse.C,
    )


def batched_coordinates(point_clouds):
    """Replacement for ME.utils.batched_coordinates.
    Takes a list of coordinate tensors and prepends batch indices.
    Args:
        point_clouds: list of Tensors, each [N_i, 3]
    Returns:
        Tensor [N_total, 4] with (batch_idx, x, y, z)
    """
    coords_list = []
    for i, pc in enumerate(point_clouds):
        if isinstance(pc, np.ndarray):
            pc = torch.from_numpy(pc)
        pc = pc.float()
        batch_ids = torch.full((pc.shape[0], 1), i, dtype=pc.dtype, device=pc.device)
        coords_list.append(torch.cat([batch_ids, pc], dim=1))
    return torch.cat(coords_list, dim=0)


def sparse_quantize(coordinates, quantization_size):
    """Replacement for ME.utils.sparse_quantize.
    Quantizes coordinates and removes duplicates.
    Args:
        coordinates: ndarray or Tensor [N, 3]
        quantization_size: float voxel size
    Returns:
        Tensor [M, 3] of unique quantized coordinates
    """
    if isinstance(coordinates, np.ndarray):
        coordinates = torch.from_numpy(coordinates).float()
    else:
        coordinates = coordinates.float()
    quantized = torch.floor(coordinates / quantization_size).int()
    # Remove duplicates
    unique, inverse = torch.unique(quantized, dim=0, return_inverse=True)
    return unique
