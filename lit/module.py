try:
    import pytorch_lightning as pl
except ImportError as exc:  # pragma: no cover - depends on environment
    raise ImportError("PyTorch Lightning is required for SCAModule.") from exc

from lit.losses import SCALoss
from lit.models import build_models
from lit.optim import configure_sca_optimizers


class SCAModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(vars(args))
        self.model, self.modelq = build_models(args)
        self.loss_fn = SCALoss(args)
        self.automatic_optimization = False

    def forward(self, data_dict):
        feats_ground = self.modelq(data_dict, mode="q")
        feats_aerial = self.model(data_dict, mode="db")
        return feats_ground, feats_aerial

    def training_step(self, batch, batch_idx):
        data_dict, triplets_local_indexes, _ = batch
        opt_db, opt_q = self.optimizers()
        local_batch_size = triplets_local_indexes.shape[0] // self.args.negs_num_per_query

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

    def validation_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        return configure_sca_optimizers(self.model, self.modelq, self.args)

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
