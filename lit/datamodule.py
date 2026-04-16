import logging

import torch
from torch.utils.data import DataLoader, Dataset

try:
    import pytorch_lightning as pl
except ImportError as exc:  # pragma: no cover - depends on environment
    raise ImportError("PyTorch Lightning is required for SCADataModule.") from exc

from datasets.datasets_ws_kitti360 import (
    KITTI360BaseDataset,
    KITTI360TripletsDataset,
    kitti360_collate_fn,
)
from datasets.datasets_ws_nuscenes import (
    NuScenesBaseDataset,
    NuScenesTripletsDataset,
    nuscenes_collate_fn,
)


class _ValidationSignalDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return torch.tensor(0)


class SCADataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.triplets_ds = None
        self.test_ds = None
        self.collate_fn = None

    def setup(self, stage=None):
        if self.triplets_ds is not None and self.test_ds is not None:
            return

        if self.args.dataset == "kitti360":
            self.triplets_ds = KITTI360TripletsDataset(
                self.args,
                self.args.datasets_folder,
                self.args.dataset_name,
                "train",
                self.args.negs_num_per_query,
            )
            self.test_ds = KITTI360BaseDataset(
                self.args, self.args.datasets_folder, self.args.dataset_name, "test"
            )
            self.collate_fn = kitti360_collate_fn
        elif self.args.dataset == "nuscenes":
            self.triplets_ds = NuScenesTripletsDataset(
                self.args,
                self.args.datasets_folder,
                self.args.dataset_name,
                "train",
                self.args.negs_num_per_query,
            )
            self.test_ds = NuScenesBaseDataset(
                self.args, self.args.datasets_folder, self.args.dataset_name, "test"
            )
            self.collate_fn = nuscenes_collate_fn
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        self.triplets_ds.triplets_global_indexes = torch.zeros(
            (max(1, self.args.cache_refresh_rate), self.args.negs_num_per_query + 2),
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
            batch_size=self.args.train_batch_size,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=str(self.args.device).startswith("cuda"),
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(_ValidationSignalDataset(), batch_size=1)
