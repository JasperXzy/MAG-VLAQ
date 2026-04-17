import sys
from pathlib import Path
import torch
import torch.nn as nn
from sca.models.layers.sparse_utils import SimpleSparse
import logging

# Add Utonia to python path dynamically
utonia_path = Path(__file__).resolve().parents[4] / "demo" / "Utonia"
utonia_path = str(utonia_path)
if utonia_path not in sys.path:
    sys.path.append(utonia_path)

from utonia.model import PointTransformerV3, load as utonia_load
from utonia.structure import Point
from utonia.utils import offset2batch


class UtoniaFE(nn.Module):
    """
    Wrapper for Utonia PointTransformerV3 as Feature Extractor.
    Supports loading pretrained weights and selective freezing.

    Input: 9-channel features (xyz + rgb + normal), matching Utonia's
    pretrained interface. RGB is projected from 2D camera images;
    points without valid projection use zeros (handled by Utonia's
    Causal Modality Blinding).
    """
    def __init__(self, out_channels=256, planes=(64, 128, 256), args=None):
        super().__init__()
        if args is None:
            raise ValueError("UtoniaFE requires explicit args; parse CLI/config in the entrypoint.")
        self.args = args

        self.planes = planes
        pretrained_name = getattr(self.args, 'utonia_pretrained', 'none')
        freeze_mode = getattr(self.args, 'unfreeze_utonia_mode', 'frozen')
        extract_stages = getattr(self.args, 'utonia_extract_stages', '1_2_3')
        self.target_stages = [int(s) for s in extract_stages.split('_')]

        if pretrained_name != 'none':
            # Load pretrained checkpoint to get architecture config
            ckpt = utonia_load(pretrained_name, ckpt_only=True)
            pretrained_config = ckpt["config"]

            # Override for our use case
            pretrained_config["enc_mode"] = True
            pretrained_config["traceable"] = True
            pretrained_config["enable_flash"] = True
            # Progressive patch size: local attention in shallow stages, global in deep stages
            pretrained_config["enc_patch_size"] = [256, 256, 512, 512, 1024]
            # Moderate drop_path for fine-tuning regularization (pretrained default is 0.3)
            pretrained_config["drop_path"] = 0.1

            self.ptv3 = PointTransformerV3(**pretrained_config)

            # Load pretrained weights (full, including original 9ch embedding)
            model_state = self.ptv3.state_dict()
            pretrained_state = ckpt["state_dict"]
            filtered_state = {k: v for k, v in pretrained_state.items()
                              if k in model_state and v.shape == model_state[k].shape}
            self.ptv3.load_state_dict(filtered_state, strict=False)
            logging.info(f"Utonia pretrained loaded. Loaded: {len(filtered_state)}/{len(pretrained_state)} keys, "
                         f"Skipped (shape mismatch or missing): {len(pretrained_state) - len(filtered_state)}")

            # Read enc_channels from pretrained config
            enc_channels = list(pretrained_config["enc_channels"])
        else:
            # From scratch: 9ch input (xyz + rgb + normal)
            enc_channels = [48, 96, 192, 384, 512]
            self.ptv3 = PointTransformerV3(
                in_channels=9,
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

            # lrutonia=0 overrides freeze_mode → fully freeze PTv3 so no
            # grad is computed and the optimizer gate in train.py drops it.
            lrutonia = getattr(self.args, 'lrutonia', 0.0)
            if lrutonia == 0.0:
                for param in self.ptv3.parameters():
                    param.requires_grad = False
                logging.info("[Utonia] lrutonia=0 -> PTv3 fully frozen (no grad)")

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
        - feat: FloatTensor of [N, 3] (unused, overwritten internally)
        - rgb: FloatTensor of [N, 3] (projected RGB, 0-1; zeros where unavailable)
        - normal: FloatTensor of [N, 3] (precomputed per-point normals)
        - offset: IntTensor or LongTensor of [B]
        """
        grid_coord = data_dict['grid_coord'].clone()
        coord = data_dict['coord'].clone()
        offset = data_dict['offset']
        rgb = data_dict.get('rgb', torch.zeros_like(coord))
        normals = data_dict.get('normal', torch.zeros_like(coord))
        batch = offset2batch(offset)

        if not getattr(self, '_logged_once', False):
            with torch.no_grad():
                c = data_dict['coord']
                g = data_dict['grid_coord']
                logging.debug(
                    f"[Utonia in] coord range=[{c.min().item():.2f},{c.max().item():.2f}] "
                    f"mean={c.mean().item():.2f} std={c.std().item():.2f}  "
                    f"grid range=[{g.min().item()},{g.max().item()}]  "
                    f"N={c.shape[0]} B={len(data_dict['offset'])}"
                )
            self._logged_once = True

        # Per-batch grid_coord shift only (PTv3 serialization expects
        # non-negative grid indices). coord must NOT be shifted: the
        # Velodyne frame already has ego at the origin, and that is the
        # distribution the Utonia ckpt was pretrained with.
        for b in range(len(offset)):
            mask = batch == b
            if mask.sum() > 0:
                grid_coord[mask] = grid_coord[mask] - grid_coord[mask].min(dim=0)[0]

        # 9ch feat keys = (coord, color, normal). coord is already in the
        # official space (meters * UTONIA_GLOBAL_SCALE) from collate.
        feat_xyz = coord
        data_dict['grid_coord'] = grid_coord
        data_dict['coord'] = coord
        data_dict['feat'] = torch.cat([feat_xyz, rgb, normals], dim=1)
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
