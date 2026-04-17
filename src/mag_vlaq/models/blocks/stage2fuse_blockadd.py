import torch
import torch.nn.functional as F
from torch import nn

from mag_vlaq.models.layers.eca_block import ECABasicBlock
from mag_vlaq.models.layers.pooling import MinkGeM
from mag_vlaq.models.layers.sparse_utils import (
    SimpleSparse,
    sparse_broadcast_add,
    sparse_global_avg_pool,
)


class SparseLinear(nn.Module):
    """Wraps nn.Linear to operate on SimpleSparse (pointwise linear transform)."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return SimpleSparse(features=self.linear(x.F), coordinates=x.C)


def select_act(act):
    if act is None:
        out = nn.Identity()
    elif act == "relu":
        out = nn.ReLU()
    elif act == "tanh":
        out = nn.Tanh()
    elif act == "sigmoid":
        out = nn.Sigmoid()
    else:
        raise NotImplementedError
    return out


class BasicBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class Basic(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out += identity
        out = self.relu(out)
        return out


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: [b, c, h, w]
        assert len(x.shape) == 4
        x = nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)
        assert len(x.shape) == 4
        return x


class FFNFuse(nn.Module):
    def __init__(self, dim, stg2fuse_type):
        super().__init__()
        self.stg2fuse_type = stg2fuse_type.split("_")
        self.ffns = nn.ModuleList()
        for e in self.stg2fuse_type:
            if e == "basic":
                self.ffns.append(Basic(dim))
            else:
                raise NotImplementedError

    def forward(self, x):
        outlist = []
        for ffn in self.ffns:
            out = ffn(x)
            outlist.append(out)
        outsum = sum(outlist)
        return outsum


class Stage2FuseBlockAdd(nn.Module):
    def __init__(self, fusedim, imgdim, bevdim, voxdim, args=None):
        super().__init__()
        if args is None:
            raise ValueError("Stage2FuseBlockAdd requires explicit args; parse CLI/config in the entrypoint.")
        self.args = args

        self.projsfusebev = nn.ModuleList()
        self.projsfuseimg = nn.ModuleList()
        self.projsfusevox = nn.ModuleList()
        self.ffnsbev = nn.ModuleList()
        self.ffnsimg = nn.ModuleList()
        self.ffnsvox = nn.ModuleList()
        self.projsbevfuse = nn.ModuleList()
        self.projsimgfuse = nn.ModuleList()
        self.projsvoxfuse = nn.ModuleList()
        self.ffnsfuse = nn.ModuleList()
        for _i in range(self.args.stg2nlayers):
            if self.args.stg2_useproj:
                self.projsfuseimg.append(nn.Sequential(nn.Linear(fusedim, imgdim)))
                self.projsfusevox.append(nn.Sequential(nn.Linear(fusedim, voxdim)))
                self.projsimgfuse.append(nn.Sequential(nn.Conv2d(imgdim, fusedim, kernel_size=1)))
                self.projsvoxfuse.append(SparseLinear(voxdim, fusedim))
            else:
                self.projsfusebev.append(nn.Identity())
                self.projsfuseimg.append(nn.Identity())
                self.projsfusevox.append(nn.Identity())
                self.projsbevfuse.append(nn.Identity())
                self.projsimgfuse.append(nn.Identity())
                self.projsvoxfuse.append(nn.Identity())
            self.ffnsimg.append(BasicBlock(imgdim))
            self.ffnsvox.append(ECABasicBlock(voxdim, voxdim))
            self.ffnsfuse.append(FFNFuse(dim=fusedim, stg2fuse_type=self.args.stg2fuse_type))

        # self.poolbev = GeM()
        self.poolimage = GeM()
        self.poolvox = MinkGeM()
        self.poolfuse = GeM()

    def forward_imgvox(self, imgmap, bevmap, voxmap, fusevec):
        # imagefeatmap: [b, c, h, w]
        # bevfeatmap: [b, c, h, w]
        # fusevec: [b, c]
        for i in range(self.args.stg2nlayers):
            projfuseimg = self.projsfuseimg[i]
            projfusevox = self.projsfusevox[i]
            ffnimg = self.ffnsimg[i]
            ffnvox = self.ffnsvox[i]
            projimgfuse = self.projsimgfuse[i]
            projvoxfuse = self.projsvoxfuse[i]
            ffnfuse = self.ffnsfuse[i]

            if self.args.stg2_type == "full":
                fusevec_img = projfuseimg(fusevec)
                fusevec_vox = projfusevox(fusevec)
                imgmap = imgmap + fusevec_img.unsqueeze(-1).unsqueeze(-1)
                voxmap = sparse_broadcast_add(voxmap, fusevec_vox)

                imgmap = ffnimg(imgmap)
                # bevmap = ffnbev(bevmap)
                voxmap = ffnvox(voxmap)
                imgoutvec = self.poolimage(imgmap).flatten(1)
                voxoutvec = self.poolvox(voxmap).flatten(1)

                if self.args.stg2fuse_type is not None:
                    imgmap_fuse = projimgfuse(imgmap)
                    voxmap_fuse = projvoxfuse(voxmap)
                    imgvec_fuse = F.adaptive_avg_pool2d(imgmap_fuse, [1, 1]).squeeze(-1).squeeze(-1)
                    voxvec_fuse = sparse_global_avg_pool(voxmap_fuse)
                    fusevec = fusevec + imgvec_fuse + voxvec_fuse
                    fusevec = ffnfuse(fusevec)

            else:
                raise NotImplementedError

        return fusevec, imgoutvec, None, voxoutvec

    def forward(self, imagemap, bevmap, voxmap, fusevec, type):
        if type != "vox":
            raise NotImplementedError
        return self.forward_imgvox(imagemap, bevmap, voxmap, fusevec)
