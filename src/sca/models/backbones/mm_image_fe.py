import torch
import torch.nn as nn


class ImageFE(nn.Module):
    def __init__(self, fe_type, layers=None, args=None):
        super().__init__()
        self.fe_type = fe_type
        if self.fe_type != 'dinov2_vitl14':
            raise NotImplementedError(
                f"Query ImageFE now only supports 'dinov2_vitl14'; got {self.fe_type}"
            )

        if args is None:
            raise ValueError("ImageFE requires explicit args; parse CLI/config in the entrypoint.")
        self.args = args

        self.fe = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.last_dim = 1024
        self.target_blocks = [int(e) for e in self.args.dino_extract_blocks.split('_')]

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
        h, w = x.shape[-2] // 14, x.shape[-1] // 14
        x_tokens = self.fe.prepare_tokens_with_masks(x)
        last_block_idx = len(self.fe.blocks) - 1
        outputs = []
        for i, blk in enumerate(self.fe.blocks):
            x_tokens = blk(x_tokens)
            if i in self.target_blocks:
                tokens = self.fe.norm(x_tokens) if i == last_block_idx else x_tokens
                patch_tokens = tokens[:, 1:]
                b, n, c = patch_tokens.shape
                assert h * w == n, f"Patch count mismatch: {h}*{w} != {n}"
                feat_map = patch_tokens.transpose(1, 2).reshape(b, c, h, w)
                outputs.append(feat_map)
        return outputs

    def forward(self, x):
        x_list = self.forward_dino(x)
        feat_map = x_list[-1]
        return feat_map, x_list
