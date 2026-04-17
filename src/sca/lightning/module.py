import logging
import math

import torch

try:
    import pytorch_lightning as pl
except ImportError as exc:  # pragma: no cover - depends on environment
    raise ImportError("PyTorch Lightning is required for SCAModule.") from exc

from sca.lightning.evaluation import assign_features, compute_recall, feature_store_length
from sca.lightning.losses import SCALoss
from sca.lightning.models import build_models
from sca.lightning.optim import configure_sca_optimizers


class SCAModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg.to_dict())
        self.model, self.modelq = build_models(cfg)
        self.loss_fn = SCALoss(cfg)
        self.automatic_optimization = False
        self.best_r1r5r10ep = [0.0, 0.0, 0.0, 0]
        self._val_db_outputs = []
        self._val_q_outputs = []

    def forward(self, data_dict):
        feats_ground = self.modelq(data_dict, mode="q")
        feats_aerial = self.model(data_dict, mode="db")
        return feats_ground, feats_aerial

    def training_step(self, batch, batch_idx):
        data_dict, triplets_local_indexes, _ = batch
        opt_db, opt_q = self.optimizers()
        local_batch_size = triplets_local_indexes.shape[0] // self.cfg.negs_num_per_query

        feats_ground, feats_aerial = self(data_dict)
        losses = self.loss_fn(
            feats_ground=feats_ground,
            feats_aerial=feats_aerial,
            data_dict=data_dict,
            triplets_local_indexes=triplets_local_indexes,
        )
        loss = losses["total"]

        opt_db.zero_grad(set_to_none=True)
        opt_q.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        self._clip_if_configured(opt_db)
        self._clip_if_configured(opt_q)
        opt_db.step()
        opt_q.step()

        self.log_dict(
            {
                "train/loss": loss.detach(),
                "train/triplet": losses["triplet"].detach(),
                "train/other": losses["other"].detach(),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.trainer.world_size > 1,
            batch_size=local_batch_size,
        )
        return loss.detach()

    def on_validation_epoch_start(self):
        self._val_db_outputs = []
        self._val_q_outputs = []
        self.cfg.device = str(self.device)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data_dict, indices = batch
        data_dict = self._move_tensor_values(data_dict)
        indices = indices.to(device=self.device, dtype=torch.long)

        if dataloader_idx == 0:
            features = self.model(data_dict, mode="db")["embedding"].float()
            self._val_db_outputs.append((indices.detach(), features.detach()))
            return {"split": "db", "indices": indices.detach(), "features": features.detach()}

        features = self.modelq(data_dict, mode="q")["embedding"].float()
        if self.cfg.test_method == "five_crops" and features.shape[0] != indices.shape[0]:
            features = torch.stack(torch.split(features, 5)).mean(1)
        self._val_q_outputs.append((indices.detach(), features.detach()))
        return {"split": "q", "indices": indices.detach(), "features": features.detach()}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        data_dict, indices = batch
        data_dict = self._move_tensor_values(data_dict)
        indices = indices.to(device=self.device, dtype=torch.long)
        if dataloader_idx == 0:
            features = self.model(data_dict, mode="db")["embedding"].float()
            split = "db"
        else:
            features = self.modelq(data_dict, mode="q")["embedding"].float()
            split = "q"
        return {
            "split": split,
            "indices": indices.detach(),
            "features": features.detach(),
        }

    def on_validation_epoch_end(self):
        test_ds = getattr(self.trainer.datamodule, "test_ds", None)
        if test_ds is None:
            return

        db_indices, db_features = self._cat_validation_outputs(self._val_db_outputs)
        q_indices, q_features = self._cat_validation_outputs(self._val_q_outputs)
        db_indices = self._all_gather_variable(db_indices)
        db_features = self._all_gather_variable(db_features)
        q_indices = self._all_gather_variable(q_indices)
        q_features = self._all_gather_variable(q_features)

        all_features = torch.zeros(
            (feature_store_length(test_ds, self.cfg.test_method), self.cfg.features_dim),
            device=self.device,
            dtype=torch.float32,
        )
        assign_features(all_features, db_indices, db_features, test_ds, "hard_resize")
        assign_features(all_features, q_indices, q_features, test_ds, self.cfg.test_method)

        all_features_np = all_features.detach().cpu().numpy()
        queries_features = all_features_np[test_ds.database_num:]
        database_features = all_features_np[:test_ds.database_num]
        recalls, _ = compute_recall(
            self.cfg,
            queries_features,
            database_features,
            test_ds,
            self.cfg.test_method,
        )
        metrics = {
            f"val/R@{value}": float(recalls[index])
            for index, value in enumerate(self.cfg.recall_values)
        }
        tracked = [
            metrics.get("val/R@1", 0.0),
            metrics.get("val/R@5", 0.0),
            metrics.get("val/R@10", 0.0),
        ]
        metrics["val/R_sum"] = sum(tracked)
        self.log_dict(metrics, prog_bar=True, rank_zero_only=False)
        self._log_retrieval_summary(tracked)

    def configure_optimizers(self):
        return configure_sca_optimizers(self.model, self.modelq, self.cfg)

    def _move_tensor_values(self, data_dict):
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(self.device)
        return data_dict

    def _cat_validation_outputs(self, outputs):
        if not outputs:
            return (
                torch.empty(0, device=self.device, dtype=torch.long),
                torch.empty(0, self.cfg.features_dim, device=self.device),
            )
        indices = torch.cat([item[0] for item in outputs], dim=0).to(self.device)
        features = torch.cat([item[1] for item in outputs], dim=0).to(self.device)
        return indices, features

    def _all_gather_variable(self, tensor):
        if getattr(self.trainer, "world_size", 1) <= 1:
            return tensor

        length = torch.tensor([tensor.shape[0]], device=self.device, dtype=torch.long)
        lengths = self.all_gather(length).reshape(-1)
        max_length = int(lengths.max().item())
        if max_length == 0:
            return tensor.new_empty((0, *tensor.shape[1:]))

        if tensor.shape[0] < max_length:
            padding = tensor.new_zeros((max_length - tensor.shape[0], *tensor.shape[1:]))
            tensor = torch.cat([tensor, padding], dim=0)

        gathered = self.all_gather(tensor)
        chunks = [
            gathered[rank, : int(lengths[rank].item())]
            for rank in range(gathered.shape[0])
            if int(lengths[rank].item()) > 0
        ]
        if not chunks:
            return tensor.new_empty((0, *tensor.shape[1:]))
        return torch.cat(chunks, dim=0)

    def _log_retrieval_summary(self, current):
        real_epoch = int(self.trainer.current_epoch) // math.ceil(
            self.cfg.queries_per_epoch / self.cfg.cache_refresh_rate
        )
        if sum(current) > sum(self.best_r1r5r10ep[:3]):
            self.best_r1r5r10ep = [*current, real_epoch]

        now = (
            f"Now : R@1 = {current[0]:.1f}   R@5 = {current[1]:.1f}   "
            f"R@10 = {current[2]:.1f}   epoch = {real_epoch:d}"
        )
        best = (
            f"Best: R@1 = {self.best_r1r5r10ep[0]:.1f}   "
            f"R@5 = {self.best_r1r5r10ep[1]:.1f}   "
            f"R@10 = {self.best_r1r5r10ep[2]:.1f}   "
            f"epoch = {self.best_r1r5r10ep[3]:d}"
        )
        if self.trainer.is_global_zero:
            logging.info(now)
            logging.info(best)

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
