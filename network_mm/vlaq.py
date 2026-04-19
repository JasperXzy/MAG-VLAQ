import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.sparse_utils import SimpleSparse


def is_vlaq_only(args):
    final_type = getattr(args, 'final_type', '')
    if isinstance(final_type, str):
        return final_type == 'vlaq_only'
    return 'vlaq_only' in final_type or ('vlaq' in final_type and 'only' in final_type)


def concat_dense_sparse(dense_tokens, sparse_tokens):
    """Append dense per-batch tokens to a SimpleSparse token set."""
    if dense_tokens.dim() != 3:
        raise ValueError(f"dense_tokens must be [B, N, C], got {dense_tokens.shape}")
    if dense_tokens.shape[-1] != sparse_tokens.F.shape[-1]:
        raise ValueError(
            f"channel mismatch: dense={dense_tokens.shape[-1]} sparse={sparse_tokens.F.shape[-1]}"
        )

    b, n, c = dense_tokens.shape
    dense_features = dense_tokens.reshape(b * n, c)
    batch_ids = torch.arange(b, device=dense_tokens.device).repeat_interleave(n)
    token_ids = torch.arange(n, device=dense_tokens.device).repeat(b)
    zeros = torch.zeros_like(token_ids)
    dense_coords = torch.stack((batch_ids, token_ids, zeros, zeros), dim=1)
    dense_coords = dense_coords.to(dtype=sparse_tokens.C.dtype)
    sparse_coords = sparse_tokens.C.to(device=dense_tokens.device)

    return SimpleSparse(
        features=torch.cat((dense_features, sparse_tokens.F), dim=0),
        coordinates=torch.cat((dense_coords, sparse_coords), dim=0),
    )


class VLAQ(nn.Module):
    def __init__(
        self,
        n_queries,
        query_dim,
        token_dim,
        dropout=0.0,
        norm=True,
        out_dim=None,
        q_init='orthogonal',
    ):
        super().__init__()
        self.n_queries = n_queries
        self.query_dim = query_dim
        self.token_dim = token_dim
        self.norm = norm
        self.scale = math.sqrt(query_dim)

        self.token_proj = (
            nn.Identity() if token_dim == query_dim else nn.Linear(token_dim, query_dim)
        )
        self.q_k = nn.Parameter(torch.empty(n_queries, query_dim))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = (
            nn.Identity() if out_dim is None else nn.Linear(n_queries * query_dim, out_dim)
        )
        self._init_queries(q_init)

    def _init_queries(self, q_init):
        if q_init == 'orthogonal':
            nn.init.orthogonal_(self.q_k)
        elif q_init == 'xavier':
            nn.init.xavier_normal_(self.q_k)
        elif q_init == 'kmeans':
            raise NotImplementedError("vlaq_q_init='kmeans' is scheduled for Phase 4")
        else:
            raise ValueError(f"Unknown vlaq_q_init: {q_init}")

    def _effective_queries(self, batch_size, q_bias=None, device=None, dtype=None):
        q = self.q_k.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        if q_bias is None:
            return q
        if q_bias.shape != q.shape:
            raise ValueError(f"q_bias must be {q.shape}, got {q_bias.shape}")
        return q + q_bias

    def _finish(self, residual):
        if self.norm:
            residual = F.normalize(residual, p=2, dim=-1)
        residual = self.dropout(residual)
        out = residual.flatten(1)
        out = self.out_proj(out)
        return F.normalize(out, p=2, dim=-1)

    def forward(self, tokens, q_bias=None):
        if isinstance(tokens, SimpleSparse):
            return self._forward_sparse(tokens, q_bias=q_bias)
        return self._forward_dense(tokens, q_bias=q_bias)

    def _forward_dense(self, tokens, q_bias=None):
        if tokens.dim() != 3:
            raise ValueError(f"dense VLAQ tokens must be [B, N, C], got {tokens.shape}")
        z = self.token_proj(tokens)
        b = z.shape[0]
        q_eff = self._effective_queries(
            b, q_bias=q_bias, device=z.device, dtype=z.dtype
        )
        scores = torch.einsum('bnc,bsc->bns', z, q_eff) / self.scale
        alpha = torch.softmax(scores, dim=1)
        residual = torch.einsum('bns,bnc->bsc', alpha, z) - q_eff
        return self._finish(residual)

    def _forward_sparse(self, tokens, q_bias=None):
        z = self.token_proj(tokens.F)
        batch_idx = tokens.batch_indices.to(device=z.device)
        batch_size = q_bias.shape[0] if q_bias is not None else tokens.num_batches
        q_eff = self._effective_queries(
            batch_size, q_bias=q_bias, device=z.device, dtype=z.dtype
        )

        q_per_token = q_eff[batch_idx]
        scores = (z.unsqueeze(1) * q_per_token).sum(dim=-1) / self.scale

        index = batch_idx.unsqueeze(1).expand(-1, self.n_queries)
        max_scores = torch.full(
            (batch_size, self.n_queries),
            -torch.inf,
            device=z.device,
            dtype=z.dtype,
        )
        max_scores.scatter_reduce_(0, index, scores, reduce='amax', include_self=True)

        exp_scores = torch.exp(scores - max_scores[batch_idx])
        denom = torch.zeros(
            batch_size, self.n_queries, device=z.device, dtype=z.dtype
        )
        denom.scatter_add_(0, index, exp_scores)
        alpha = exp_scores / denom[batch_idx].clamp_min(torch.finfo(z.dtype).eps)

        source = alpha.unsqueeze(-1) * (z.unsqueeze(1) - q_per_token)
        residual = torch.zeros(
            batch_size, self.n_queries, self.query_dim, device=z.device, dtype=z.dtype
        )
        residual.scatter_add_(
            0,
            batch_idx.view(-1, 1, 1).expand(-1, self.n_queries, self.query_dim),
            source,
        )
        return self._finish(residual)
