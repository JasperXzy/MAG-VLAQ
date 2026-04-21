import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaQ(nn.Module):
    """Low-rank query bias generator for ODE-conditioned VLAQ."""

    def __init__(
        self,
        C,
        S,
        D,
        r=64,
        alpha_init=0.0,
        alpha_learn=True,
        layer_norm=True,
    ):
        super().__init__()
        if C <= 0 or S <= 0 or D <= 0 or r <= 0:
            raise ValueError(f"DeltaQ dims must be positive, got C={C}, S={S}, D={D}, r={r}")
        self.C = C
        self.S = S
        self.D = D
        self.r = r

        self.norm = (
            nn.LayerNorm(C, elementwise_affine=False) if layer_norm else nn.Identity()
        )
        self.w_down = nn.Linear(C, r)
        self.w_up_q = nn.Linear(r, S * D)
        if alpha_learn:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha_init)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_down.weight)
        nn.init.zeros_(self.w_down.bias)
        nn.init.zeros_(self.w_up_q.weight)
        nn.init.zeros_(self.w_up_q.bias)

    def forward(self, e_fuse):
        if e_fuse.dim() != 2:
            raise ValueError(f"e_fuse must be [B, C], got {e_fuse.shape}")
        if e_fuse.shape[-1] != self.C:
            raise ValueError(f"e_fuse channel dim must be {self.C}, got {e_fuse.shape[-1]}")

        h = F.gelu(self.w_down(self.norm(e_fuse)))
        q_bias = self.w_up_q(h).view(e_fuse.shape[0], self.S, self.D)
        alpha = self.alpha.to(device=q_bias.device, dtype=q_bias.dtype)
        return alpha * q_bias
