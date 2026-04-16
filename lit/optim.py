import logging
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn

from models_baseline.dbvanilla2d import DBVanilla2D
from network_mm.mm import MM


def _trainable(parameters: Iterable[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    return [p for p in parameters if p.requires_grad]


def _add_group(groups: List[dict], params: Iterable[torch.nn.Parameter], lr: float):
    params = list(params)
    if params:
        groups.append({"params": params, "lr": lr})


def build_param_groups(model: nn.Module, modelq: nn.Module, args) -> Tuple[List[dict], List[dict]]:
    params_db: List[dict] = []
    if isinstance(model, DBVanilla2D):
        if getattr(args, "lrdino", 0.0) > 0.0:
            dino_params = list(model.dbimage_fes.parameters())
            base_params = [
                p for name, p in model.named_parameters()
                if not name.startswith("dbimage_fes.")
            ]
            _add_group(params_db, _trainable(dino_params), args.lrdino)
            _add_group(params_db, _trainable(base_params), args.lrdb)
        else:
            _add_group(params_db, _trainable(model.parameters()), args.lrdb)

    params_q: List[dict] = []
    if isinstance(modelq, MM):
        if getattr(args, "lrdino", 0.0) > 0.0:
            _add_group(params_q, _trainable(modelq.image_fe.parameters()), args.lrdino)
        else:
            _add_group(params_q, _trainable(modelq.image_fe.parameters()), args.lr)

        _add_group(params_q, _trainable(modelq.image_pool.parameters()), args.lr)

        utonia_ptv3 = _trainable(modelq.vox_fe.ptv3.parameters())
        utonia_proj = _trainable(modelq.vox_fe.projs.parameters())
        if getattr(args, "lrutonia", 0.0) > 0.0:
            _add_group(params_q, utonia_ptv3, args.lrutonia)
        _add_group(params_q, utonia_proj, args.lrpc)

        other_vox = [
            p for name, p in modelq.vox_fe.named_parameters()
            if p.requires_grad
            and not name.startswith("ptv3.")
            and not name.startswith("projs.")
        ]
        _add_group(params_q, other_vox, args.lrpc)
        _add_group(params_q, _trainable(modelq.vox_pool.parameters()), args.lrpc)
        _add_group(params_q, _trainable(modelq.fuseblocktoshallow.parameters()), args.lr)
        _add_group(params_q, _trainable(modelq.stg2fuseblock.parameters()), args.lr)
        _add_group(params_q, _trainable(modelq.stg2fusefc.parameters()), args.lr)

        _add_group(params_q, _trainable(modelq.image_proj.parameters()), args.lr)
        _add_group(params_q, _trainable(modelq.vox_proj.parameters()), args.lrpc)
        _add_group(params_q, _trainable(modelq.stg2image_proj.parameters()), args.lr)
        _add_group(params_q, _trainable(modelq.stg2vox_proj.parameters()), args.lrpc)

        scalar_groups = [
            (modelq.image_weight, args.lr),
            (modelq.vox_weight, args.lrpc),
            (modelq.shallow_weight, args.lr),
            (modelq.imageorg_weight, args.lr),
            (modelq.voxorg_weight, args.lr),
            (modelq.shalloworg_weight, args.lr),
            (modelq.stg2image_weight, args.lr),
            (modelq.stg2vox_weight, args.lrpc),
            (modelq.stg2fuse_weight, args.lr),
        ]
        for param, lr in scalar_groups:
            _add_group(params_q, [param], lr)

    if not params_db:
        raise RuntimeError("No trainable parameters found for the database model.")
    if not params_q:
        raise RuntimeError("No parameters found for the query model optimizer.")
    return params_db, params_q


def configure_sca_optimizers(model: nn.Module, modelq: nn.Module, args):
    params_db, params_q = build_param_groups(model, modelq, args)
    optimizer_db = torch.optim.Adam(params_db)
    optimizer_q = torch.optim.Adam(params_q)
    _log_optimizer_summary(optimizer_db, optimizer_q, modelq)
    return [optimizer_db, optimizer_q]


def _log_optimizer_summary(optimizer_db, optimizer_q, modelq):
    num_params_db = sum(
        p.numel() for group in optimizer_db.param_groups for p in group["params"]
    )
    num_params_q = sum(
        p.numel() for group in optimizer_q.param_groups for p in group["params"]
    )
    logging.info("Number of parameters in optimizerdb: %d", num_params_db)
    logging.info("Number of parameters in optimizerq: %d", num_params_q)
    for i, group in enumerate(optimizer_q.param_groups):
        n_params = sum(p.numel() for p in group["params"])
        logging.info("[param_group %d] %.3fM params @ lr=%s", i, n_params / 1e6, group["lr"])

    if hasattr(modelq, "vox_fe") and hasattr(modelq.vox_fe, "ptv3"):
        ptv3 = modelq.vox_fe.ptv3
        n_total = sum(p.numel() for p in ptv3.parameters())
        n_train = sum(p.numel() for p in ptv3.parameters() if p.requires_grad)
        logging.info("[PTv3] trainable %.2fM / total %.2fM", n_train / 1e6, n_total / 1e6)
