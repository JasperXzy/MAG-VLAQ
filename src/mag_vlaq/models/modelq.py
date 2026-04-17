import torch
import torch.nn.functional as F
from torch import nn

from mag_vlaq.models.backbones.modelq_image_fe import ImageFE
from mag_vlaq.models.backbones.modelq_image_pooling import GeM
from mag_vlaq.models.backbones.modelq_utonia_fe import UtoniaFE
from mag_vlaq.models.blocks.fuse_block_toshallow import FuseBlockToShallow
from mag_vlaq.models.blocks.stage2fuse_blockadd import Stage2FuseBlockAdd
from mag_vlaq.models.layers.pooling import MinkGeM


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


class ModelQ(nn.Module):
    def __init__(self, drop=None, args=None):
        super().__init__()
        if args is None:
            raise ValueError("ModelQ requires explicit args; parse CLI/config in the entrypoint.")
        self.args = args
        self.drop = drop
        # ---- query
        self.image_fe = ImageFE(fe_type="dinov2_vitl14", args=self.args)
        self.image_pool = GeM()
        planes = [int(x) for x in self.args.mm_voxfe_planes.split("_")]
        self.planes = planes
        self.vox_fe = UtoniaFE(out_channels=planes[-1], planes=planes, args=self.args)
        self.vox_pool = MinkGeM()

        # DINOv2 returns one feature map; repeat it to match Utonia's multi-stage fusion inputs.
        img_dims = [self.args.mm_imgfe_dim for _ in range(len(planes))]

        # Ensure img_dims are correctly passed to FuseBlockToShallow
        self.fuseblocktoshallow = FuseBlockToShallow(
            dims=[self.args.mm_stg2fuse_dim for _ in range(len(planes))],
            img_dims=img_dims,
            vox_dims=planes,
            bev_dims=[int(e) for e in self.args.mm_bevfe_planes.split("_")],
            args=self.args,
        )
        self.stg2fuseblock = Stage2FuseBlockAdd(
            fusedim=self.args.mm_stg2fuse_dim,
            imgdim=self.args.mm_imgfe_dim,
            bevdim=self.args.mm_bevfe_dim,
            voxdim=self.args.mm_voxfe_dim,
            args=self.args,
        )
        self.stg2fusefc = nn.Linear(self.args.mm_stg2fuse_dim, self.args.mm_stg2fuse_dim)

        self.image_weight = nn.Parameter(
            torch.tensor(self.args.image_weight, dtype=torch.float32), requires_grad=self.args.image_learnweight
        )
        self.vox_weight = nn.Parameter(
            torch.tensor(self.args.vox_weight, dtype=torch.float32), requires_grad=self.args.vox_learnweight
        )
        self.shallow_weight = nn.Parameter(
            torch.tensor(self.args.shallow_weight, dtype=torch.float32), requires_grad=self.args.shallow_learnweight
        )

        self.imageorg_weight = nn.Parameter(
            torch.tensor(self.args.imagevoxorg_weight, dtype=torch.float32),
            requires_grad=self.args.imagevoxorg_learnweight,
        )
        self.voxorg_weight = nn.Parameter(
            torch.tensor(self.args.imagevoxorg_weight, dtype=torch.float32),
            requires_grad=self.args.imagevoxorg_learnweight,
        )
        self.shalloworg_weight = nn.Parameter(
            torch.tensor(self.args.shalloworg_weight, dtype=torch.float32),
            requires_grad=self.args.shalloworg_learnweight,
        )

        self.stg2image_weight = nn.Parameter(
            torch.tensor(self.args.stg2imagevox_weight, dtype=torch.float32),
            requires_grad=self.args.stg2imagevox_learnweight,
        )
        self.stg2vox_weight = nn.Parameter(
            torch.tensor(self.args.stg2imagevox_weight, dtype=torch.float32),
            requires_grad=self.args.stg2imagevox_learnweight,
        )
        self.stg2fuse_weight = nn.Parameter(
            torch.tensor(self.args.stg2fuse_weight, dtype=torch.float32), requires_grad=self.args.stg2fuse_learnweight
        )

        self.image_proj = MLP(self.args.mm_imgfe_dim, self.args.features_dim)

        self.vox_proj = MLP(planes[-1], self.args.features_dim)
        self.stg2image_proj = MLP(self.args.mm_imgfe_dim, self.args.features_dim)
        self.stg2vox_proj = MLP(self.args.mm_voxfe_dim, self.args.features_dim)

    # ====  query
    def forward_q(self, data_dict):
        if self.drop == "image":
            data_dict["query_image"] = data_dict["query_image"] * 0
        elif self.drop == "pc":
            data_dict["coords"][:, 1:] = data_dict["coords"][:, 1:] * 0

        image = data_dict["query_image"]
        output = []
        if "image" in self.args.output_type:
            imagefeatmap, imagefeatmaplist = self.image_fe(image)
            imagefeatmaplist = [imagefeatmap for _ in range(len(self.planes))]

            imagefeatvec = self.image_pool(imagefeatmap)
            imagefeatvec = imagefeatvec.flatten(1)
            imagefeatvec = self.image_proj(imagefeatvec)
            if self.args.output_l2 is True:
                imagefeatvec = F.normalize(imagefeatvec, dim=-1)
            imagefeatvec_org = imagefeatvec
            output.append(imagefeatvec * self.image_weight)
        if "vox" in self.args.output_type:
            utonia_dict = {
                "coord": data_dict["utonia_coord"],
                "grid_coord": data_dict["utonia_grid_coord"],
                "feat": data_dict["utonia_feat"],
                "rgb": data_dict["utonia_rgb"],
                "normal": data_dict["utonia_normal"],
                "offset": data_dict["utonia_offset"],
            }
            voxfeatmap, voxfeatmaplist = self.vox_fe(utonia_dict)
            voxfeatvec = self.vox_pool(voxfeatmap)
            voxfeatvec = self.vox_proj(voxfeatvec)
            if self.args.output_l2 is True:
                voxfeatvec = F.normalize(voxfeatvec, dim=-1)
            voxfeatvec_org = voxfeatvec
            output.append(voxfeatvec * self.vox_weight)

        # ==== stage-1 fusion, ME
        if "shallow" in self.args.output_type:
            if "vox" in self.args.output_type:
                shallowfeatvec = self.fuseblocktoshallow(imagefeatmaplist, None, voxfeatmaplist, type="vox")
            shallowfeatvecorg = shallowfeatvec
            if self.args.output_l2 is True:
                shallowfeatvec = F.normalize(shallowfeatvec, dim=-1)
            output.append(shallowfeatvec * self.shallow_weight)
        elif "addorg" in self.args.output_type:
            if "vox" in self.args.output_type:
                addorgvec = imagefeatvec_org + voxfeatvec_org
            shallowfeatvecorg = addorgvec
            if self.args.output_l2 is True:
                addorgvec = F.normalize(addorgvec, dim=-1)
            output.append(addorgvec * self.shallow_weight)

        # ==== stage-2 fusion, ME
        if "vox" in self.args.output_type:
            stg2fusevec, stg2imagevec, stg2bevvec, stg2voxvec = self.stg2fuseblock(
                imagefeatmap, None, voxfeatmap, output[-1], type="vox"
            )
            stg2imagevec = self.stg2image_proj(stg2imagevec)
            stg2voxvec = self.stg2vox_proj(stg2voxvec)

        stg2fusevec = self.stg2fusefc(stg2fusevec)

        # ==== final output
        finaloutput = []
        if "imageorg" in self.args.final_type:
            finaloutput.append(imagefeatvec_org * self.imageorg_weight)
        if "voxorg" in self.args.final_type:
            finaloutput.append(voxfeatvec_org * self.voxorg_weight)
        if "shalloworg" in self.args.final_type:
            finaloutput.append(shallowfeatvec * self.shalloworg_weight)
        if "stg2image" in self.args.final_type:
            finaloutput.append(stg2imagevec * self.stg2image_weight)
        if "stg2vox" in self.args.final_type:
            finaloutput.append(stg2voxvec * self.stg2vox_weight)
        if "stg2fuse" in self.args.final_type:
            finaloutput.append(stg2fusevec * self.stg2fuse_weight)

        if self.args.final_fusetype == "add":
            x = sum(finaloutput)
        elif self.args.final_fusetype == "cat":
            x = torch.cat(finaloutput, dim=-1)
        elif self.args.final_fusetype == "catadd":
            x = torch.cat(finaloutput[:-1], dim=-1)
            x = x + finaloutput[-1]

        if self.args.final_l2 is True:
            x = F.normalize(x, dim=-1)

        output_dict = {
            "imagevec_org": imagefeatvec_org,
            "voxvec_org": voxfeatvec_org,
            "shallowvec_org": shallowfeatvecorg,
            "stg2fusevec": stg2fusevec,
            "stg2imagevec": stg2imagevec,
            "stg2voxvec": stg2voxvec,
            "embedding": x,
        }

        return output_dict

    def forward(self, data_dict, mode):
        # resize img
        if mode == "q":
            x = self.forward_q(data_dict)
        else:
            raise NotImplementedError

        return x
