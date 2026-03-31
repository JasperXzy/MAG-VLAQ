import sys
import os
import torch
import torch.nn as nn
import MinkowskiEngine as ME

# Add Utonia to python path dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
utonia_path = os.path.join(os.path.dirname(current_dir), 'demo', 'Utonia')
if utonia_path not in sys.path:
    sys.path.append(utonia_path)

from utonia.model import PointTransformerV3
from utonia.structure import Point
from utonia.utils import offset2batch

class UtoniaFE(nn.Module):
    """
    Wrapper for Utonia PointTransformerV3 to act as a Feature Extractor (FE).
    It intercepts the multi-stage outputs of PTv3 and converts them to 
    MinkowskiEngine.SparseTensor format for compatibility with existing fusion blocks.
    """
    def __init__(self, in_channels=1, out_channels=256, planes=(64, 128, 256)):
        super().__init__()
        
        self.planes = planes
        
        # PTv3 is configured with its default deeper 5-stage architecture
        # Fix: enc_channels[-1] is 576 so head_dim is 576/24 = 24 (divisible by 3 for 3D RoPE)
        self.ptv3 = PointTransformerV3(
            in_channels=in_channels,
            order=["z", "z-trans", "hilbert", "hilbert-trans"],
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(48, 96, 192, 384, 576),
            enc_num_head=(2, 4, 8, 16, 24),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3, # Common drop path rate
            shuffle_orders=True,
            pre_norm=True,
            enable_rpe=False,
            enable_flash=False, # Set to True if flash attention is installed and supported
            upcast_attention=False,
            upcast_softmax=False,
            mask_token=False,
            traceable=True, # VERY IMPORTANT: Allows tracing back pooling parents
            enc_mode=True,  # Encoder-only mode, we don't need point-wise predictions
        )
        
        # Target stages mapping (e.g. 192, 384, 576)
        self.target_stages = [2, 3, 4] # Corresponds to depths generating 192, 384, 576
        self.utonia_channels = [192, 384, 576]
        
        # Projection layers to map Utonia's native channels to the requested `planes`
        from MinkowskiEngine import MinkowskiLinear
        self.projs = nn.ModuleList([
            MinkowskiLinear(self.utonia_channels[i], planes[i]) for i in range(len(planes))
        ])
        
    def forward(self, data_dict):
        """
        data_dict must contain:
        - coord: FloatTensor of [N, 3]
        - grid_coord: IntTensor of [N, 3]
        - feat: FloatTensor of [N, C]
        - offset: IntTensor or LongTensor of [B]
        """
        # 1. Initialize Utonia Point structure with per-batch centered coords
        # This prevents Z-order curve serialization overflow from huge absolute global coords
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
        point = Point(data_dict)
        
        # 2. Forward pass through PointTransformerV3's encoder
        # PTv3 returns the deepest level `Point` object when enc_mode=True
        final_point = self.ptv3(point)
        
        # 3. Retrieve representations from pooled stages by tracking "pooling_parent"
        stages = []
        curr_point = final_point
        for i in range(5): # Maximum 5 stages
            stages.append(curr_point)
            if "pooling_parent" in curr_point:
                curr_point = curr_point.pop("pooling_parent")
            else:
                break
                
        # stages current order: [stage4, stage3, stage2, stage1, stage0] (deepest to shallowest)
        # Reverse to get [stage0, stage1, stage2, stage3, stage4] (shallowest to deepest)
        stages = stages[::-1]
        
        # Target stages mapping (e.g. 192, 384, 576)
        selected_stages = [stages[i] for i in self.target_stages]
        
        # 4. Convert selected points to ME.SparseTensor and map dimensions
        tensor_list = []
        for i, p in enumerate(selected_stages):
            batch = offset2batch(p.offset)
            # ME requires coordinates in format [Batch, X, Y, Z]
            # Utonia's p.grid_coord is already spatially quantized
            coords = torch.cat([batch.unsqueeze(-1).to(p.grid_coord.device), p.grid_coord], dim=1).int()
            
            sp_tensor = ME.SparseTensor(features=p.feat, coordinates=coords)
            # Project to match requested plane dimensions
            sp_tensor = self.projs[i](sp_tensor)
            tensor_list.append(sp_tensor)
            
        # 5. Return the deepest tensor, and the list of multi-scale tensors.
        # This precisely matches the API of MinkFPN.
        voxfeatmap = tensor_list[-1]
        voxfeatmaplist = tensor_list
        return voxfeatmap, voxfeatmaplist
