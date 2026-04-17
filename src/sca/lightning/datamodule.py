import logging

import torch
from torch.utils.data import DataLoader, Subset

try:
    import pytorch_lightning as pl
except ImportError as exc:  # pragma: no cover - depends on environment
    raise ImportError("PyTorch Lightning is required for SCADataModule.") from exc

from sca.data.kitti360 import (
    KITTI360BaseDataset,
    KITTI360TripletsDataset,
    kitti360_collate_fn_cache_db,
    kitti360_collate_fn_cache_q,
    kitti360_collate_fn,
)
from sca.data.nuscenes import (
    NuScenesBaseDataset,
    NuScenesTripletsDataset,
    nuscenes_collate_fn_cache_db,
    nuscenes_collate_fn_cache_q,
    nuscenes_collate_fn,
)


def _worker_kwargs(cfg, num_workers):
    kwargs = {"num_workers": num_workers}
    context = getattr(cfg, "worker_multiprocessing_context", None)
    if num_workers > 0 and context not in (None, "", "default"):
        kwargs["multiprocessing_context"] = context
    return kwargs


class SCADataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.triplets_ds = None
        self.test_ds = None
        self.collate_fn = None

    def setup(self, stage=None):
        if self.triplets_ds is not None and self.test_ds is not None:
            return

        if self.cfg.dataset == "kitti360":
            self.triplets_ds = KITTI360TripletsDataset(
                self.cfg,
                self.cfg.datasets_folder,
                self.cfg.dataset_name,
                "train",
                self.cfg.negs_num_per_query,
            )
            self.test_ds = KITTI360BaseDataset(
                self.cfg, self.cfg.datasets_folder, self.cfg.dataset_name, "test"
            )
            self.collate_fn = kitti360_collate_fn
        elif self.cfg.dataset == "nuscenes":
            self.triplets_ds = NuScenesTripletsDataset(
                self.cfg,
                self.cfg.datasets_folder,
                self.cfg.dataset_name,
                "train",
                self.cfg.negs_num_per_query,
            )
            self.test_ds = NuScenesBaseDataset(
                self.cfg, self.cfg.datasets_folder, self.cfg.dataset_name, "test"
            )
            self.collate_fn = nuscenes_collate_fn
        else:
            raise ValueError(f"Unsupported dataset: {self.cfg.dataset}")

        self.triplets_ds.triplets_global_indexes = torch.zeros(
            (max(1, self.cfg.cache_refresh_rate), self.cfg.negs_num_per_query + 2),
            dtype=torch.long,
        )
        logging.info("Train query set: %s", self.triplets_ds)
        logging.info("Test set: %s", self.test_ds)

    def train_dataloader(self):
        if self.triplets_ds is None:
            raise RuntimeError("SCADataModule.setup() must run before train_dataloader().")
        self.triplets_ds.is_inference = False
        return DataLoader(
            self.triplets_ds,
            batch_size=self.cfg.train_batch_size,
            collate_fn=self.collate_fn,
            pin_memory=str(self.cfg.device).startswith("cuda"),
            drop_last=True,
            **_worker_kwargs(self.cfg, self.cfg.num_workers),
        )

    def val_dataloader(self):
        if self.test_ds is None:
            raise RuntimeError("SCADataModule.setup() must run before val_dataloader().")

        self.test_ds.test_method = self.cfg.test_method
        collate_db, collate_q = self._eval_collate_fns()
        database_ds = Subset(self.test_ds, range(self.test_ds.database_num))
        queries_ds = Subset(
            self.test_ds,
            range(
                self.test_ds.database_num,
                self.test_ds.database_num + self.test_ds.queries_num,
            ),
        )
        query_batch_size = (
            1 if self.cfg.test_method == "single_query" else self.cfg.infer_batch_size
        )
        loader_kwargs = {
            "num_workers": self.cfg.num_workers,
            "pin_memory": str(self.cfg.device).startswith("cuda"),
            "shuffle": False,
        }
        loader_kwargs.update(_worker_kwargs(self.cfg, self.cfg.num_workers))
        return [
            DataLoader(
                database_ds,
                batch_size=self.cfg.infer_batch_size,
                collate_fn=collate_db,
                **loader_kwargs,
            ),
            DataLoader(
                queries_ds,
                batch_size=query_batch_size,
                collate_fn=collate_q,
                **loader_kwargs,
            ),
        ]

    def _eval_collate_fns(self):
        if self.cfg.dataset == "kitti360":
            return kitti360_collate_fn_cache_db, kitti360_collate_fn_cache_q
        if self.cfg.dataset == "nuscenes":
            return nuscenes_collate_fn_cache_db, nuscenes_collate_fn_cache_q
        raise ValueError(f"Unsupported dataset: {self.cfg.dataset}")
