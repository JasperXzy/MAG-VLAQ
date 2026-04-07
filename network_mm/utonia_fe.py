import sys
import os
import torch
import torch.nn as nn
from layers.sparse_utils import SimpleSparse
import logging

# Add Utonia to python path dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
utonia_path = os.path.join(os.path.dirname(current_dir), 'demo', 'Utonia')
if utonia_path not in sys.path:
    sys.path.append(utonia_path)

from utonia.model import PointTransformerV3, load as utonia_load
from utonia.structure import Point
from utonia.utils import offset2batch


class UtoniaFE(nn.Module):
    """
    Wrapper for Utonia PointTransformerV3 as Feature Extractor.
    Supports loading pretrained weights and selective freezing.

    Input: 3-channel features (xyz coordinates).
    For pretrained models (originally 9ch: xyz+rgb+normal), the embedding
    layer is surgically adapted to 3ch by retaining the xyz columns of
    the pretrained weight matrix.
    """
    def __init__(self, out_channels=256, planes=(64, 128, 256)):
        super().__init__()
        from tools.options import parse_arguments
        opt = parse_arguments()

        self.planes = planes
        pretrained_name = getattr(opt, 'utonia_pretrained', 'none')
        freeze_mode = getattr(opt, 'unfreeze_utonia_mode', 'frozen')
        extract_stages = getattr(opt, 'utonia_extract_stages', '1_2_3')
        self.target_stages = [int(s) for s in extract_stages.split('_')]

        if pretrained_name != 'none':
            # Load pretrained checkpoint to get architecture config
            ckpt = utonia_load(pretrained_name, ckpt_only=True)
            pretrained_config = ckpt["config"]

            # Override for our use case
            pretrained_config["enc_mode"] = True
            pretrained_config["traceable"] = True
            pretrained_config["enable_flash"] = True
            # Reduce patch size to lower attention memory: O(patch_size^2)
            pretrained_config["enc_patch_size"] = [1024 for _ in range(len(pretrained_config.get("enc_patch_size", [1024]*5)))]
            # Set drop_path=0 for stable fine-tuning (pretrained default is 0.3)
            pretrained_config["drop_path"] = 0.0

            self.ptv3 = PointTransformerV3(**pretrained_config)

            # Load pretrained weights (full, including original embedding)
            model_state = self.ptv3.state_dict()
            pretrained_state = ckpt["state_dict"]
            filtered_state = {k: v for k, v in pretrained_state.items()
                              if k in model_state and v.shape == model_state[k].shape}
            self.ptv3.load_state_dict(filtered_state, strict=False)
            logging.info(f"Utonia pretrained loaded. Loaded: {len(filtered_state)}/{len(pretrained_state)} keys, "
                         f"Skipped (shape mismatch or missing): {len(pretrained_state) - len(filtered_state)}")

            # Adapt embedding: pretrained in_channels (e.g. 9: xyz+rgb+normal) → 3 (xyz only)
            # Retain only the first 3 columns (xyz) of the pretrained embedding weight
            old_linear = self.ptv3.embedding.stem.linear  # nn.Linear(9, embed_ch)
            pretrained_in_ch = old_linear.in_features
            embed_ch = old_linear.out_features
            new_linear = nn.Linear(3, embed_ch)
            with torch.no_grad():
                new_linear.weight.copy_(old_linear.weight[:, :3])
                new_linear.bias.copy_(old_linear.bias)
            self.ptv3.embedding.stem.linear = new_linear
            self.ptv3.embedding.in_channels = 3
            logging.info(f"Utonia embedding adapted: {pretrained_in_ch}ch → 3ch (xyz only), "
                         f"retained xyz columns of pretrained weight [{embed_ch} output dims]")

            # Read enc_channels from pretrained config
            enc_channels = list(pretrained_config["enc_channels"])
        else:
            # From scratch: 3ch input (xyz coordinates)
            enc_channels = [48, 96, 192, 384, 512]
            self.ptv3 = PointTransformerV3(
                in_channels=3,
                order=["z", "z-trans", "hilbert", "hilbert-trans"],
                stride=(2, 2, 2, 2),
                enc_depths=(2, 2, 2, 6, 2),
                enc_channels=enc_channels,
                enc_num_head=(2, 4, 8, 16, 16),
                enc_patch_size=(256, 256, 256, 256, 256),
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                attn_drop=0.0,
                proj_drop=0.0,
                drop_path=0.1,
                shuffle_orders=True,
                pre_norm=True,
                enable_rpe=False,
                enable_flash=False,
                upcast_attention=False,
                upcast_softmax=False,
                mask_token=False,
                traceable=True,
                enc_mode=True,
            )

        # Freeze/unfreeze logic
        if pretrained_name != 'none':
            lrutonia = getattr(opt, 'lrutonia', 0.0)
            if lrutonia == 0.0:
                freeze_mode = 'frozen'

            # Freeze all first
            for param in self.ptv3.parameters():
                param.requires_grad = False

            num_stages = len(self.ptv3.enc)
            if freeze_mode == 'full':
                for param in self.ptv3.parameters():
                    param.requires_grad = True
            elif freeze_mode == 'last1':
                # Unfreeze last 1 encoder stage only
                stage_name = f"enc{num_stages - 1}"
                if hasattr(self.ptv3.enc, stage_name):
                    for param in getattr(self.ptv3.enc, stage_name).parameters():
                        param.requires_grad = True

            # Always keep the adapted embedding trainable for domain adaptation
            for param in self.ptv3.embedding.parameters():
                param.requires_grad = True

            n_total = sum(p.numel() for p in self.ptv3.parameters())
            n_trainable = sum(p.numel() for p in self.ptv3.parameters() if p.requires_grad)
            logging.info(f"Utonia freeze_mode={freeze_mode}: {n_trainable}/{n_total} params trainable "
                         f"({100*n_trainable/n_total:.1f}%)")

        # Target stage channels from enc_channels
        self.utonia_channels = [enc_channels[s] for s in self.target_stages]

        # Projection layers to map Utonia's native channels to the requested planes
        self.projs = nn.ModuleList([
            nn.Linear(self.utonia_channels[i], planes[i]) for i in range(len(planes))
        ])

    def forward(self, data_dict):
        """
        data_dict must contain:
        - coord: FloatTensor of [N, 3]
        - grid_coord: IntTensor of [N, 3]
        - feat: FloatTensor of [N, 3] (raw xyz, centered here before embedding)
        - offset: IntTensor or LongTensor of [B]
        """
        # 1. Per-batch centering of coord, grid_coord, and feat
        #    feat = xyz coordinates, centered to match the pretrained data distribution
        #    where feat[:3] == coord (both centered relative coords)
        grid_coord = data_dict['grid_coord'].clone()
        coord = data_dict['coord'].clone()
        offset = data_dict['offset']
        batch = offset2batch(offset)

        for b in range(len(offset)):
            mask = batch == b
            if mask.sum() > 0:
                grid_min = grid_coord[mask].min(dim=0)[0]
                grid_coord[mask] = grid_coord[mask] - grid_min
                coord_min = coord[mask].min(dim=0)[0]
                coord[mask] = coord[mask] - coord_min

        data_dict['grid_coord'] = grid_coord
        data_dict['coord'] = coord
        # Use centered coord as feat: consistent with pretrained PTv3 where feat[:3] == coord
        data_dict['feat'] = coord.clone()
        point = Point(data_dict)

        # 2. Forward pass through PTv3 encoder
        final_point = self.ptv3(point)

        # 3. Retrieve representations from pooled stages via pooling_parent chain
        stages = []
        curr_point = final_point
        for i in range(5):
            stages.append(curr_point)
            if "pooling_parent" in curr_point:
                curr_point = curr_point.pop("pooling_parent")
            else:
                break

        # Reverse: [stage4,...,stage0] → [stage0,...,stage4]
        stages = stages[::-1]

        selected_stages = [stages[i] for i in self.target_stages]

        # 4. Convert selected points to ME.SparseTensor and project dimensions
        tensor_list = []
        for i, p in enumerate(selected_stages):
            batch = offset2batch(p.offset)
            coords = torch.cat([batch.unsqueeze(-1).to(p.grid_coord.device), p.grid_coord], dim=1).int()

            sp_tensor = SimpleSparse(features=p.feat, coordinates=coords)
            sp_tensor = SimpleSparse(features=self.projs[i](sp_tensor.F), coordinates=sp_tensor.C)
            tensor_list.append(sp_tensor)

        voxfeatmap = tensor_list[-1]
        voxfeatmaplist = tensor_list
        return voxfeatmap, voxfeatmaplist
