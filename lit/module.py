import logging
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn

try:
    import pytorch_lightning as pl
except ImportError as exc:  # pragma: no cover - depends on environment
    raise ImportError("PyTorch Lightning is required for SCAModule.") from exc

from compute_other_loss import compute_other_loss
from layers.lora import is_lora_param_name, is_adapter_param_name
from models_baseline.dbvanilla2d import DBVanilla2D
from network_mm.mm import MM
from network_mm.vlaq import VLAQ, is_vlaq_only


def _trainable(parameters: Iterable[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    return [p for p in parameters if p.requires_grad]


def _add_group(groups: List[dict], params: Iterable[torch.nn.Parameter], lr: float):
    params = list(params)
    if params:
        groups.append({"params": params, "lr": lr})


def _split_peft(named_params):
    """Given an iterable of (name, param), split trainable params into
    (lora, adapter, other) buckets. LoRA params have 'lora_A'/'lora_B' in
    their name; adapter params live under a child module named 'adapter'."""
    lora_trainable: List[torch.nn.Parameter] = []
    adapter_trainable: List[torch.nn.Parameter] = []
    other_trainable: List[torch.nn.Parameter] = []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        if is_lora_param_name(name):
            lora_trainable.append(p)
        elif is_adapter_param_name(name):
            adapter_trainable.append(p)
        else:
            other_trainable.append(p)
    return lora_trainable, adapter_trainable, other_trainable


def _add_image_fe_groups(groups, image_fe_module, args, default_lr):
    """Split an ImageFE's trainable params into LoRA / MultiConv-adapter /
    non-PEFT groups. Each gets its own LR.

    - LoRA params      -> lrdino_lora (default args.lr)
    - Adapter params   -> lrdino_mc (default args.lrpc)
    - Other trainable  -> lrdino if >0, else default_lr
    """
    lora_p, adapter_p, other_p = _split_peft(image_fe_module.named_parameters())
    if lora_p:
        lrdino_lora = getattr(args, "lrdino_lora", None)
        lr = float(lrdino_lora) if lrdino_lora is not None else args.lr
        _add_group(groups, lora_p, lr)
    if adapter_p:
        lrdino_mc = getattr(args, "lrdino_mc", None)
        lr = float(lrdino_mc) if lrdino_mc is not None else getattr(args, "lrpc", args.lr)
        _add_group(groups, adapter_p, lr)
    if other_p:
        base_lr = args.lrdino if getattr(args, "lrdino", 0.0) > 0.0 else default_lr
        _add_group(groups, other_p, base_lr)


def _add_utonia_fe_groups(groups, vox_fe_module, args):
    """Split UtoniaFE's trainable params into PTv3 LoRA / MLP-adapter /
    PTv3 non-PEFT (e.g. embedding, last1) / projs / other_vox groups.
    Handles lora / mlp_adapter / last1 / frozen / full modes uniformly."""
    ptv3 = vox_fe_module.ptv3
    ptv3_lora, ptv3_adapter, ptv3_other = _split_peft(ptv3.named_parameters())

    if ptv3_lora:
        lrutonia_lora = getattr(args, "lrutonia_lora", None)
        lr = float(lrutonia_lora) if lrutonia_lora is not None else args.lrpc
        _add_group(groups, ptv3_lora, lr)

    if ptv3_adapter:
        lrutonia_mlp = getattr(args, "lrutonia_mlp", None)
        lr = float(lrutonia_mlp) if lrutonia_mlp is not None else args.lrpc
        _add_group(groups, ptv3_adapter, lr)

    if ptv3_other and getattr(args, "lrutonia", 0.0) > 0.0:
        _add_group(groups, ptv3_other, args.lrutonia)

    _add_group(groups, _trainable(vox_fe_module.projs.parameters()), args.lrpc)

    other_vox = [
        p for name, p in vox_fe_module.named_parameters()
        if p.requires_grad
        and not name.startswith("ptv3.")
        and not name.startswith("projs.")
    ]
    _add_group(groups, other_vox, args.lrpc)


def build_param_groups(model: nn.Module, modelq: nn.Module, args) -> Tuple[List[dict], List[dict]]:
    vlaq_only = is_vlaq_only(args)
    params_db: List[dict] = []
    if isinstance(model, DBVanilla2D):
        if vlaq_only:
            _add_image_fe_groups(params_db, model.dbimage_fes, args, default_lr=args.lrdb)
            if getattr(model, "chart_a", None) is not None:
                _add_group(params_db, _trainable(model.chart_a.parameters()), args.lrdb)
        elif getattr(args, "lrdino", 0.0) > 0.0 or getattr(args, "unfreeze_dino_mode", "frozen") in ("lora", "multiconv"):
            _add_image_fe_groups(params_db, model.dbimage_fes, args, default_lr=args.lrdb)
            base_params = [
                p for name, p in model.named_parameters()
                if not name.startswith("dbimage_fes.")
            ]
            _add_group(params_db, _trainable(base_params), args.lrdb)
        else:
            _add_group(params_db, _trainable(model.parameters()), args.lrdb)

    params_q: List[dict] = []
    if isinstance(modelq, MM):
        _add_image_fe_groups(params_q, modelq.image_fe, args, default_lr=args.lr)

        _add_group(params_q, _trainable(modelq.image_pool.parameters()), args.lr)

        _add_utonia_fe_groups(params_q, modelq.vox_fe, args)
        _add_group(params_q, _trainable(modelq.vox_pool.parameters()), args.lrpc)
        if getattr(modelq, "fuseblocktoshallow", None) is not None:
            _add_group(params_q, _trainable(modelq.fuseblocktoshallow.parameters()), args.lr)
        if getattr(modelq, "stg2fuseblock", None) is not None:
            _add_group(params_q, _trainable(modelq.stg2fuseblock.parameters()), args.lr)
        if getattr(modelq, "stg2fusefc", None) is not None:
            _add_group(params_q, _trainable(modelq.stg2fusefc.parameters()), args.lr)

        _add_group(params_q, _trainable(modelq.image_proj.parameters()), args.lr)
        _add_group(params_q, _trainable(modelq.vox_proj.parameters()), args.lrpc)
        if getattr(modelq, "stg2image_proj", None) is not None:
            _add_group(params_q, _trainable(modelq.stg2image_proj.parameters()), args.lr)
        if getattr(modelq, "stg2vox_proj", None) is not None:
            _add_group(params_q, _trainable(modelq.stg2vox_proj.parameters()), args.lrpc)

        if vlaq_only:
            if getattr(modelq, "chart_img", None) is not None:
                _add_group(params_q, _trainable(modelq.chart_img.parameters()), args.lr)
            if getattr(modelq, "chart_vox_l", None) is not None:
                _add_group(params_q, _trainable(modelq.chart_vox_l.parameters()), args.lrpc)
            elif getattr(modelq, "chart_vox", None) is not None:
                _add_group(params_q, _trainable(modelq.chart_vox.parameters()), args.lrpc)
            if getattr(modelq, "delta_q", None) is not None:
                _add_group(
                    params_q,
                    _trainable(modelq.delta_q.parameters()),
                    args.lr if getattr(args, "lrodecq", None) is None else args.lrodecq,
                )
            if getattr(modelq, "vlaq", None) is not None:
                _add_group(params_q, _trainable(modelq.vlaq.parameters()), args.lr)
        else:
            scalar_groups = [
                (modelq.image_weight, args.lr),
                (modelq.vox_weight, args.lrpc),
                (modelq.shallow_weight, args.lr),
                (modelq.imageorg_weight, args.lr),
                (modelq.voxorg_weight, args.lr),
                (modelq.shalloworg_weight, args.lr),
                (modelq.stg2image_weight, args.lr),
                (modelq.stg2vox_weight, args.lr),
                (modelq.stg2fuse_weight, args.lr),
            ]
            for param, lr in scalar_groups:
                _add_group(params_q, [param], lr)

    if not params_db:
        raise RuntimeError("No trainable parameters found for the database model.")
    if not params_q:
        raise RuntimeError("No parameters found for the query model optimizer.")
    return params_db, params_q


def compute_triplet_loss(args, criterion_triplet, triplets_local_indexes, features):
    loss_triplet = features.new_tensor(0.0)
    local_batch = triplets_local_indexes.shape[0] // args.negs_num_per_query
    triplets_local_indexes = triplets_local_indexes.view(
        local_batch, args.negs_num_per_query, 3
    ).transpose(1, 0)

    for triplets in triplets_local_indexes:
        queries_indexes, positives_indexes, negatives_indexes = triplets.T
        loss_triplet = loss_triplet + criterion_triplet(
            features[queries_indexes],
            features[positives_indexes],
            features[negatives_indexes],
        )

    return loss_triplet / (local_batch * args.negs_num_per_query)


class SCAModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(vars(args))
        self.model = DBVanilla2D(mode="db", dim=args.features_dim, args=args)
        self.modelq = MM(args=args)
        self.shared_vlaq = None
        if is_vlaq_only(args):
            self._init_shared_vlaq()
        self.criterion_triplet = nn.TripletMarginLoss(
            margin=args.margin, p=2, reduction="sum"
        )
        self.automatic_optimization = False

    def _init_shared_vlaq(self):
        vlaq_out_dim = self.args.vlaq_out_dim
        embedding_dim = (
            self.args.vlaq_n_queries * self.args.vlaq_query_dim
            if vlaq_out_dim is None else vlaq_out_dim
        )
        if embedding_dim != self.args.features_dim:
            raise ValueError(
                "VLAQ embedding dim must match features_dim for triplet cache: "
                f"got vlaq={embedding_dim}, features_dim={self.args.features_dim}"
            )

        def make_vlaq():
            return VLAQ(
                n_queries=self.args.vlaq_n_queries,
                query_dim=self.args.vlaq_query_dim,
                token_dim=self.args.vlaq_token_dim,
                out_dim=vlaq_out_dim,
                dropout=self.args.vlaq_dropout,
                q_init=self.args.vlaq_q_init,
            )

        vlaq_init_seed = getattr(self.args, "vlaq_init_seed", None)
        if vlaq_init_seed is None:
            self.shared_vlaq = make_vlaq()
        else:
            rng_state = torch.random.get_rng_state()
            try:
                torch.manual_seed(int(vlaq_init_seed))
                self.shared_vlaq = make_vlaq()
            finally:
                torch.random.set_rng_state(rng_state)
        self.modelq.vlaq = self.shared_vlaq
        self.model.shared_vlaq = self.shared_vlaq
        assert id(self.modelq.vlaq.q_k) == id(self.model.shared_vlaq.q_k)

    def forward(self, data_dict):
        feats_ground = self.modelq(data_dict, mode="q")
        feats_aerial = self.model(data_dict, mode="db")
        return feats_ground, feats_aerial

    def training_step(self, batch, batch_idx):
        data_dict, triplets_local_indexes, _ = batch
        opt_db, opt_q = self.optimizers()
        local_batch_size = triplets_local_indexes.shape[0] // self.args.negs_num_per_query

        feats_ground, feats_aerial = self(data_dict)
        other_loss = compute_other_loss(
            feats_ground,
            feats_aerial,
            data_dict,
            positive_thd=self.args.train_positives_dist_threshold,
            negative_thd=self.args.val_positive_dist_threshold,
            args=self.args,
        )

        feats_ground_embed = feats_ground["embedding"].unsqueeze(1)
        feats_aerial_embed = feats_aerial["embedding"]
        features = torch.cat((feats_ground_embed, feats_aerial_embed), dim=1)
        features = features.view(-1, self.args.features_dim)
        triplet_loss = compute_triplet_loss(
            self.args, self.criterion_triplet, triplets_local_indexes, features
        )
        loss = other_loss + triplet_loss * self.args.tripletloss_weight

        opt_db.zero_grad(set_to_none=True)
        opt_q.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        self._clip_if_configured(opt_db)
        self._clip_if_configured(opt_q)
        opt_db.step()
        opt_q.step()

        log_values = {
            "train/loss": loss.detach(),
            "train/triplet": triplet_loss.detach(),
            "train/other": other_loss.detach(),
        }
        ode_cq_stats = getattr(self.modelq, "ode_cq_stats", None)
        if ode_cq_stats:
            log_values.update(
                {
                    f"delta_q/{key}": value.detach()
                    for key, value in ode_cq_stats.items()
                }
            )

        self.log_dict(
            log_values,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.trainer.world_size > 1,
            batch_size=local_batch_size,
        )
        return loss.detach()

    def validation_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        params_db, params_q = build_param_groups(self.model, self.modelq, self.args)
        optimizer_db = torch.optim.Adam(params_db)
        optimizer_q = torch.optim.Adam(params_q)

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

        if hasattr(self.modelq, "vox_fe") and hasattr(self.modelq.vox_fe, "ptv3"):
            ptv3 = self.modelq.vox_fe.ptv3
            n_total = sum(p.numel() for p in ptv3.parameters())
            n_train = sum(p.numel() for p in ptv3.parameters() if p.requires_grad)
            logging.info("[PTv3] trainable %.2fM / total %.2fM", n_train / 1e6, n_total / 1e6)

        # R@1-gated ReduceLROnPlateau: stepped manually from RetrievalEvalCallback
        # once R@1 first crosses r1_schedule_threshold (default 50). Skips LR
        # shrinking during phase A (q_bias still climbing to the max_ratio clamp),
        # only kicks in after the model enters the stable retrieval regime.
        scheduler_db = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_db, mode="max", factor=0.5, patience=3, min_lr=1e-7
        )
        scheduler_q = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_q, mode="max", factor=0.5, patience=3, min_lr=1e-7
        )
        return (
            [optimizer_db, optimizer_q],
            [
                {"scheduler": scheduler_db, "monitor": "val/R@1", "interval": "epoch"},
                {"scheduler": scheduler_q, "monitor": "val/R@1", "interval": "epoch"},
            ],
        )

    def maybe_step_r1_schedulers(self, r1_value):
        threshold = float(getattr(self.args, "r1_schedule_threshold", 50.0))
        if not hasattr(self, "_r1_schedule_started"):
            self._r1_schedule_started = False
        if float(r1_value) >= threshold:
            self._r1_schedule_started = True
        if not self._r1_schedule_started:
            return
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return
        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]
        for sch in schedulers:
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sch.step(float(r1_value))
            else:
                sch.step()

    def _clip_if_configured(self, optimizer):
        clip_val = getattr(self.trainer, "gradient_clip_val", None)
        if clip_val is None or clip_val == 0:
            return
        clip_algorithm = getattr(self.trainer, "gradient_clip_algorithm", "norm")
        self.clip_gradients(
            optimizer,
            gradient_clip_val=clip_val,
            gradient_clip_algorithm=clip_algorithm,
        )
