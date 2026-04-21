import torch
import torch.nn as nn
import torch.nn.functional as F

from network_mm.image_fe import ImageFE
from network_mm.image_pooling import GeM
from network_mm.chart import Chart2D, Chart3D
from network_mm.fuse_block_toshallow import FuseBlockToShallow
from network_mm.ode_cq import DeltaQ
from network_mm.stage2fuse_blockadd import Stage2FuseBlockAdd
from network_mm.vlaq import concat_dense_sparse, is_vlaq_only

from network_mm.utonia_fe import UtoniaFE
from layers.pooling import MinkGeM

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
    def forward(self, x):
        return self.seq(x)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {'1', 'true', 'yes', 'y'}
    return bool(value)

class MM(nn.Module):
    def __init__(self, drop=None, args=None):
        super().__init__()
        if args is None:
            raise ValueError("MM requires explicit args; parse CLI/config in the entrypoint.")
        self.args = args
        self.drop = drop
        # ---- query
        self.image_fe = ImageFE(fe_type='dinov2_vitl14', args=self.args)
        self.image_pool = GeM()
        planes = [int(x) for x in self.args.mm_voxfe_planes.split('_')]
        self.planes = planes
        self.vox_fe = UtoniaFE(out_channels=planes[-1], planes=planes, args=self.args)
        self.vox_pool = MinkGeM()

        # DINOv2 layer outputs share the same channel dim; stage count follows Utonia planes.
        img_dims = [self.args.mm_imgfe_dim for _ in range(len(planes))]
        self.vlaq_only = is_vlaq_only(self.args)
        self.use_ode_cq = self.vlaq_only and _as_bool(getattr(self.args, 'use_ode_cq', False))
        self.vlaq = None
        if self.vlaq_only:
            self.chart_img = Chart2D(self.args.mm_imgfe_dim, self.args.vlaq_token_dim)
            self.chart_vox = Chart3D(planes[-1], self.args.vlaq_token_dim)
            if self.use_ode_cq:
                self.chart_vox_l = nn.ModuleList(
                    [Chart3D(plane, self.args.vlaq_token_dim) for plane in planes[:-1]]
                    + [self.chart_vox]
                )
            else:
                self.chart_vox_l = None
        else:
            self.chart_img = None
            self.chart_vox = None
            self.chart_vox_l = None

        # Ensure img_dims are correctly passed to FuseBlockToShallow
        if self.vlaq_only:
            if self.use_ode_cq:
                token_dims = [self.args.vlaq_token_dim for _ in range(len(planes))]
                self.fuseblocktoshallow = FuseBlockToShallow(
                    dims=token_dims,
                    img_dims=token_dims,
                    vox_dims=token_dims,
                    bev_dims=[int(e) for e in self.args.mm_bevfe_planes.split('_')],
                    args=self.args,
                )
                self.delta_q = DeltaQ(
                    C=self.args.vlaq_token_dim,
                    S=self.args.vlaq_n_queries,
                    D=self.args.vlaq_query_dim,
                    r=self.args.ode_cq_rank,
                    alpha_init=self.args.ode_cq_alpha_init,
                    alpha_learn=_as_bool(self.args.ode_cq_alpha_learn),
                )
            else:
                self.fuseblocktoshallow = None
                self.delta_q = None
            self.stg2fuseblock = None
            self.stg2fusefc = None
        else:
            self.delta_q = None
            self.fuseblocktoshallow = FuseBlockToShallow(dims=[self.args.mm_stg2fuse_dim for _ in range(len(planes))],
                                                         img_dims=img_dims,
                                                         vox_dims=planes,
                                                         bev_dims=[int(e) for e in self.args.mm_bevfe_planes.split('_')],
                                                         args=self.args)
            self.stg2fuseblock = Stage2FuseBlockAdd(fusedim=self.args.mm_stg2fuse_dim, imgdim=self.args.mm_imgfe_dim, bevdim=self.args.mm_bevfe_dim, voxdim=self.args.mm_voxfe_dim, args=self.args)
            self.stg2fusefc = nn.Linear(self.args.mm_stg2fuse_dim, self.args.mm_stg2fuse_dim)


        self.image_weight = nn.Parameter(torch.tensor(self.args.image_weight, dtype=torch.float32), requires_grad=self.args.image_learnweight)
        self.vox_weight = nn.Parameter(torch.tensor(self.args.vox_weight, dtype=torch.float32), requires_grad=self.args.vox_learnweight)
        self.shallow_weight = nn.Parameter(torch.tensor(self.args.shallow_weight, dtype=torch.float32), requires_grad=self.args.shallow_learnweight)


        self.imageorg_weight = nn.Parameter(torch.tensor(self.args.imagevoxorg_weight, dtype=torch.float32), requires_grad=self.args.imagevoxorg_learnweight)
        self.voxorg_weight = nn.Parameter(torch.tensor(self.args.imagevoxorg_weight, dtype=torch.float32), requires_grad=self.args.imagevoxorg_learnweight)
        self.shalloworg_weight = nn.Parameter(torch.tensor(self.args.shalloworg_weight, dtype=torch.float32), requires_grad=self.args.shalloworg_learnweight)

        self.stg2image_weight = nn.Parameter(torch.tensor(self.args.stg2imagevox_weight, dtype=torch.float32), requires_grad=self.args.stg2imagevox_learnweight)
        self.stg2vox_weight = nn.Parameter(torch.tensor(self.args.stg2imagevox_weight, dtype=torch.float32), requires_grad=self.args.stg2imagevox_learnweight)
        self.stg2fuse_weight = nn.Parameter(torch.tensor(self.args.stg2fuse_weight, dtype=torch.float32), requires_grad=self.args.stg2fuse_learnweight)
        if self.vlaq_only:
            for param in (
                self.image_weight,
                self.vox_weight,
                self.shallow_weight,
                self.imageorg_weight,
                self.voxorg_weight,
                self.shalloworg_weight,
                self.stg2image_weight,
                self.stg2vox_weight,
                self.stg2fuse_weight,
            ):
                param.requires_grad_(False)

        self.image_proj = MLP(self.args.mm_imgfe_dim, self.args.features_dim)

        self.vox_proj = MLP(planes[-1], self.args.features_dim)
        if self.vlaq_only:
            self.stg2image_proj = None
            self.stg2vox_proj = None
            self.ode_cq_stats = {}
        else:
            self.stg2image_proj = MLP(self.args.mm_imgfe_dim, self.args.features_dim)
            self.stg2vox_proj = MLP(self.args.mm_voxfe_dim, self.args.features_dim)

    @staticmethod
    def _flatten_patch_tokens(feat_map):
        return feat_map.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _build_utonia_dict(data_dict):
        return {
            'coord': data_dict['utonia_coord'],
            'grid_coord': data_dict['utonia_grid_coord'],
            'feat': data_dict['utonia_feat'],
            'rgb': data_dict['utonia_rgb'],
            'normal': data_dict['utonia_normal'],
            'offset': data_dict['utonia_offset'],
        }

    def _ode_cq_tokens_and_bias(self, imagefeatmaplist, voxfeatmaplist):
        if self.fuseblocktoshallow is None or self.delta_q is None:
            raise RuntimeError("ODE-CQ path requires fuseblocktoshallow and delta_q")
        if self.chart_vox_l is None:
            raise RuntimeError("ODE-CQ path requires per-stage voxel charts")

        z_img_list = [
            self.chart_img(self._flatten_patch_tokens(feat_map))
            for feat_map in imagefeatmaplist
        ]
        z_vox_list = [
            self.chart_vox_l[i](sparse_stage)
            for i, sparse_stage in enumerate(voxfeatmaplist)
        ]
        fusedveclist = [
            self.fuseblocktoshallow.per_scale_summary(z_img_list[i], '2d', i)
            + self.fuseblocktoshallow.per_scale_summary(z_vox_list[i], '3d', i)
            for i in range(len(self.planes))
        ]
        e_fuse = self.fuseblocktoshallow.forward_state(fusedveclist)
        q_bias = self._stabilize_q_bias(self.delta_q(e_fuse))
        return z_img_list[-1], z_vox_list[-1], q_bias

    def _stabilize_q_bias(self, q_bias):
        bias_scale = float(getattr(self.args, 'ode_cq_bias_scale', 1.0))
        if bias_scale != 1.0:
            q_bias = q_bias * bias_scale

        if self.vlaq is None:
            return q_bias

        qk_norm = self.vlaq.q_k.detach().float().flatten().norm().clamp_min(1e-12)
        qbias_norm = q_bias.detach().float().flatten(1).norm(dim=1)
        max_ratio = float(getattr(self.args, 'ode_cq_max_ratio', 0.0) or 0.0)
        if max_ratio > 0.0:
            max_norm = qk_norm * max_ratio
            clamp_scale = (max_norm / qbias_norm.clamp_min(1e-12)).clamp(max=1.0)
            q_bias = q_bias * clamp_scale.to(device=q_bias.device, dtype=q_bias.dtype).view(-1, 1, 1)
            qbias_norm = q_bias.detach().float().flatten(1).norm(dim=1)

        alpha = getattr(self.delta_q, 'alpha', q_bias.new_tensor(0.0))
        self.ode_cq_stats = {
            'alpha': alpha.detach().float().mean(),
            'q_bias_norm': qbias_norm.mean(),
            'qk_norm': qk_norm,
            'qbias_to_qk_ratio': (qbias_norm / qk_norm).mean(),
        }
        return q_bias

    def forward_tokens(self, data_dict):
        """Return unpooled query-side tokens for later VLAQ consumption."""
        image = data_dict['query_image']
        if self.drop == 'image':
            image = image * 0

        _, imagefeatmaplist = self.image_fe(image)
        assert len(imagefeatmaplist) == len(self.planes), (
            f"dino_extract_blocks should output {len(self.planes)} layers, "
            f"got {len(imagefeatmaplist)}"
        )
        image_tokens_per_layer = [
            self._flatten_patch_tokens(feat_map) for feat_map in imagefeatmaplist
        ]

        _, voxfeatmaplist = self.vox_fe(self._build_utonia_dict(data_dict))
        assert len(voxfeatmaplist) == len(self.planes), (
            f"utonia_extract_stages should output {len(self.planes)} stages, "
            f"got {len(voxfeatmaplist)}"
        )
        for stage_idx, sparse_stage in enumerate(voxfeatmaplist):
            assert sparse_stage.F.shape[0] == sparse_stage.batch_indices.shape[0], (
                f"vox stage {stage_idx} has mismatched features and batch indices"
            )

        return {
            'image_tokens_per_layer': image_tokens_per_layer,
            'vox_sparse_per_stage': voxfeatmaplist,
        }


    # ====  query
    def forward_q(self, data_dict):
        if self.drop == 'image':
            data_dict['query_image'] = data_dict['query_image'] * 0
        elif self.drop == 'pc':
            data_dict['coords'][:,1:] = data_dict['coords'][:,1:] * 0
        
        image = data_dict['query_image']
        output = []
        if 'image' in self.args.output_type:
            imagefeatmap, imagefeatmaplist = self.image_fe(image)
            assert len(imagefeatmaplist) == len(self.planes), (
                f"dino_extract_blocks should output {len(self.planes)} layers, "
                f"got {len(imagefeatmaplist)}"
            )

            imagefeatvec = self.image_pool(imagefeatmap)
            imagefeatvec = imagefeatvec.flatten(1)
            imagefeatvec = self.image_proj(imagefeatvec)
            if self.args.output_l2 is True:
                imagefeatvec = F.normalize(imagefeatvec, dim=-1)
            imagefeatvec_org = imagefeatvec
            output.append(imagefeatvec * self.image_weight)
        if 'vox' in self.args.output_type:
            utonia_dict = self._build_utonia_dict(data_dict)
            voxfeatmap, voxfeatmaplist = self.vox_fe(utonia_dict)
            assert len(voxfeatmaplist) == len(self.planes), (
                f"utonia_extract_stages should output {len(self.planes)} stages, "
                f"got {len(voxfeatmaplist)}"
            )
            voxfeatvec = self.vox_pool(voxfeatmap)
            voxfeatvec = self.vox_proj(voxfeatvec)
            if self.args.output_l2 is True:
                voxfeatvec = F.normalize(voxfeatvec, dim=-1)
            voxfeatvec_org = voxfeatvec
            output.append(voxfeatvec * self.vox_weight)
            a=1

        if self.vlaq_only:
            if self.vlaq is None:
                raise RuntimeError("MM.vlaq must be injected by SCAModule for final_type=vlaq_only")
            if 'image' not in self.args.output_type or 'vox' not in self.args.output_type:
                raise ValueError("final_type=vlaq_only requires output_type to include image and vox")

            if self.use_ode_cq:
                z_img, z_vox, q_bias = self._ode_cq_tokens_and_bias(
                    imagefeatmaplist, voxfeatmaplist
                )
            else:
                img_tokens_last = self._flatten_patch_tokens(imagefeatmap)
                z_img = self.chart_img(img_tokens_last)
                z_vox = self.chart_vox(voxfeatmap)
                q_bias = None
            z_ground = concat_dense_sparse(z_img, z_vox)
            x = self.vlaq(z_ground, q_bias=q_bias)
            if self.args.final_l2 is True:
                x = F.normalize(x, dim=-1)

            return {
                'imagevec_org': imagefeatvec_org,
                'voxvec_org': voxfeatvec_org,
                'shallowvec_org': None,
                'stg2fusevec': None,
                'stg2imagevec': None,
                'stg2voxvec': None,
                'embedding': x,
            }

            
        # ==== stage-1 fusion, ME
        if 'shallow' in self.args.output_type:
            if 'vox' in self.args.output_type:
                shallowfeatvec = self.fuseblocktoshallow(imagefeatmaplist, None, voxfeatmaplist, type='vox')
            shallowfeatvecorg = shallowfeatvec
            if self.args.output_l2 is True:
                shallowfeatvec = F.normalize(shallowfeatvec, dim=-1)
            output.append(shallowfeatvec * self.shallow_weight)
        elif 'addorg' in self.args.output_type:
            if 'vox' in self.args.output_type:
                addorgvec = imagefeatvec_org + voxfeatvec_org
            shallowfeatvecorg = shallowfeatvec
            if self.args.output_l2 is True:
                addorgvec = F.normalize(addorgvec, dim=-1)
            output.append(addorgvec * self.shallow_weight)

        
        # ==== stage-2 fusion, ME
        if 'vox' in self.args.output_type:
            stg2fusevec, stg2imagevec, stg2bevvec, stg2voxvec = self.stg2fuseblock(imagefeatmap, None, voxfeatmap, output[-1],type='vox')
            stg2imagevec = self.stg2image_proj(stg2imagevec)
            stg2voxvec = self.stg2vox_proj(stg2voxvec)

        stg2fusevec = self.stg2fusefc(stg2fusevec)



        # ==== final output
        finaloutput = []
        if 'imageorg' in self.args.final_type:
            finaloutput.append(imagefeatvec_org * self.imageorg_weight)
        if 'voxorg' in self.args.final_type:
            finaloutput.append(voxfeatvec_org * self.voxorg_weight)
        if 'shalloworg' in self.args.final_type:
            finaloutput.append(shallowfeatvec * self.shalloworg_weight)
        if 'stg2image' in self.args.final_type:
            finaloutput.append(stg2imagevec * self.stg2image_weight)
        if 'stg2vox' in self.args.final_type:
            finaloutput.append(stg2voxvec * self.stg2vox_weight)
        if 'stg2fuse' in self.args.final_type:
            finaloutput.append(stg2fusevec * self.stg2fuse_weight)

        if self.args.final_fusetype == 'add':
            x = sum(finaloutput)
        elif self.args.final_fusetype == 'cat':
            x = torch.cat(finaloutput, dim=-1)
        elif self.args.final_fusetype == 'catadd':
            x = torch.cat(finaloutput[:-1], dim=-1)
            x = x + finaloutput[-1]

        if self.args.final_l2 is True:
            x = F.normalize(x, dim=-1)

        

        output_dict = {
            'imagevec_org': imagefeatvec_org,
            'voxvec_org': voxfeatvec_org,
            'shallowvec_org': shallowfeatvecorg,
            'stg2fusevec': stg2fusevec,
            'stg2imagevec': stg2imagevec,
            'stg2voxvec': stg2voxvec,
            'embedding': x
        }
        
        return output_dict




    def forward(self, data_dict, mode):
        # resize img
        if mode == 'q':
            x = self.forward_q(data_dict)
        else:
            raise NotImplementedError
        
        return x
