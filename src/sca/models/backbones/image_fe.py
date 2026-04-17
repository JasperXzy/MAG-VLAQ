import torch
import torch.nn as nn


class ImageFE(nn.Module):
    def __init__(self, fe_type, layers=None, args=None):
        super().__init__()
        self.fe_type = fe_type
        if self.fe_type != 'dinov2_vitl14':
            raise NotImplementedError(
                f"Database ImageFE now only supports 'dinov2_vitl14'; got {self.fe_type}"
            )

        if args is None:
            raise ValueError("ImageFE requires explicit args; parse CLI/config in the entrypoint.")
        self.args = args

        self.fe = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.last_dim = 1024

        dino_mode = getattr(self.args, 'unfreeze_dino_mode', 'frozen')
        if self.args.lrdino == 0.0:
            dino_mode = 'frozen'

        for param in self.fe.parameters():
            param.requires_grad = False

        if dino_mode == 'full':
            for param in self.fe.parameters():
                param.requires_grad = True
        elif dino_mode == 'last2':
            if hasattr(self.fe, 'blocks'):
                for block in self.fe.blocks[-2:]:
                    for param in block.parameters():
                        param.requires_grad = True
            if hasattr(self.fe, 'norm') and self.fe.norm is not None:
                for param in self.fe.norm.parameters():
                    param.requires_grad = True

    def forward_dino(self, x):
        out = self.fe.forward_features(x)
        patch_tokens = out["x_norm_patchtokens"]
        b, n, c = patch_tokens.shape
        h, w = x.shape[-2] // 14, x.shape[-1] // 14
        assert h * w == n, f"Patch count mismatch: {h}*{w} != {n}"
        patch_feat_map = patch_tokens.transpose(1, 2).reshape(b, c, h, w)
        return [patch_feat_map]

    def forward(self, x):
        x_list = self.forward_dino(x)
        feat_map = x_list[-1]
        return feat_map, x_list
