from contextlib import nullcontext

import faiss
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from sca.data.kitti360 import (
    kitti360_collate_fn_cache_db,
    kitti360_collate_fn_cache_q,
)
from sca.data.nuscenes import (
    nuscenes_collate_fn_cache_db,
    nuscenes_collate_fn_cache_q,
)


class SparseFeatureCache:
    """Sparse row store for descriptor caches used by partial mining."""

    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.matrix = [None] * shape[0]

    def __setitem__(self, indexes, values):
        assert values.shape[1] == self.shape[1], f"{values.shape[1]} {self.shape[1]}"
        for index, value in zip(indexes, values):
            self.matrix[int(index)] = value.astype(self.dtype, copy=False)

    def __getitem__(self, index):
        if hasattr(index, "__len__") and not isinstance(index, (str, bytes)):
            return np.array([self.matrix[int(i)] for i in index])
        return self.matrix[int(index)]


def _cache_num_workers(cfg):
    return int(getattr(cfg, "cache_num_workers", getattr(cfg, "num_workers", 0)))


def _worker_kwargs(cfg, num_workers):
    kwargs = {"num_workers": num_workers}
    context = getattr(cfg, "worker_multiprocessing_context", None)
    if num_workers > 0 and context not in (None, "", "default"):
        kwargs["multiprocessing_context"] = context
    return kwargs


def _progress(iterable, progress=None, desc=None, disable=False):
    if progress is not None and not disable:
        total = len(iterable) if hasattr(iterable, "__len__") else None
        progress("start", desc, total)
        try:
            for item in iterable:
                yield item
                progress("advance", desc, 1)
        finally:
            progress("close", desc, None)
        return

    yield from iterable


def _amp_context(cfg, device):
    if device.type != "cuda":
        return nullcontext()
    amp_dtype = getattr(cfg, "amp_dtype", "none")
    if amp_dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if amp_dtype == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _collate_fns(dataset_name):
    if dataset_name == "kitti360":
        return kitti360_collate_fn_cache_db, kitti360_collate_fn_cache_q
    if dataset_name == "nuscenes":
        return nuscenes_collate_fn_cache_db, nuscenes_collate_fn_cache_q
    raise ValueError(f"Unsupported dataset: {dataset_name}")


class TripletCacheBuilder:
    def __init__(self, cfg, trainer, pl_module, dataset):
        self.cfg = cfg
        self.trainer = trainer
        self.pl_module = pl_module
        self.dataset = dataset
        self.device = pl_module.device
        self.world_size = int(getattr(trainer, "world_size", 1))
        self.rank = int(getattr(trainer, "global_rank", 0))
        self.ddp_on = (
            self.world_size > 1
            and dist.is_available()
            and dist.is_initialized()
        )

    def refresh(self, progress=None):
        db_was_training = self.pl_module.model.training
        q_was_training = self.pl_module.modelq.training
        self.pl_module.model.eval()
        self.pl_module.modelq.eval()
        try:
            triplets = self._mine(progress)
        finally:
            self.pl_module.model.train(db_was_training)
            self.pl_module.modelq.train(q_was_training)
        return self._broadcast_triplets(triplets)

    def _mine(self, progress):
        mining = self.dataset.mining
        if mining == "random":
            return self._mine_random(progress)
        if mining == "full":
            return self._mine_full(progress)
        if mining in ("partial", "msls_weighted", "partial_sep"):
            return self._mine_partial(progress)
        raise NotImplementedError(f"Unsupported triplet mining mode: {mining}")

    def _mine_random(self, progress):
        sampled_queries = self._sample_queries(weighted=False)
        positives = self._positive_indexes(sampled_queries)
        subset = positives + list(sampled_queries + self.dataset.database_num)
        cache = self._compute_cache(Subset(self.dataset, subset), progress)

        triplets = []
        if self.rank == 0:
            for query_index in _progress(sampled_queries, progress, "mine"):
                query_features = self._query_features(query_index, cache)
                best_positive = self._best_positive_index(
                    query_index, cache, query_features
                )
                soft_positives = self.dataset.soft_positives_per_query[query_index]
                neg_indexes = np.random.choice(
                    self.dataset.database_num,
                    size=self.dataset.negs_num_per_query + len(soft_positives),
                    replace=False,
                )
                neg_indexes = np.setdiff1d(
                    neg_indexes, soft_positives, assume_unique=True
                )[: self.dataset.negs_num_per_query]
                triplets.append((query_index, best_positive, *neg_indexes))
        return self._triplets_tensor(triplets)

    def _mine_full(self, progress):
        sampled_queries = self._sample_queries(weighted=False)
        database_indexes = list(range(self.dataset.database_num))
        subset = database_indexes + list(sampled_queries + self.dataset.database_num)
        cache = self._compute_cache(Subset(self.dataset, subset), progress)

        triplets = []
        if self.rank == 0:
            for query_index in _progress(sampled_queries, progress, "mine"):
                query_features = self._query_features(query_index, cache)
                best_positive = self._best_positive_index(
                    query_index, cache, query_features
                )
                neg_indexes = np.random.choice(
                    self.dataset.database_num,
                    self.dataset.neg_samples_num,
                    replace=False,
                )
                soft_positives = self.dataset.soft_positives_per_query[query_index]
                neg_indexes = np.setdiff1d(
                    neg_indexes, soft_positives, assume_unique=True
                )
                neg_indexes = np.unique(
                    np.concatenate([self.dataset.neg_cache[query_index], neg_indexes])
                )
                neg_indexes = self._hardest_negatives(
                    cache, query_features, neg_indexes
                )
                self.dataset.neg_cache[query_index] = neg_indexes
                triplets.append((query_index, best_positive, *neg_indexes))
        return self._triplets_tensor(triplets)

    def _mine_partial(self, progress):
        sampled_queries = self._sample_queries(
            weighted=self.dataset.mining == "msls_weighted"
        )
        sampled_database = self._sample_database()
        positives = self._positive_indexes(sampled_queries)
        database_indexes = list(np.unique(list(sampled_database) + positives))
        subset = database_indexes + list(sampled_queries + self.dataset.database_num)
        cache = self._compute_cache(Subset(self.dataset, subset), progress)

        triplets = []
        if self.rank == 0:
            for query_index in _progress(sampled_queries, progress, "mine"):
                query_features = self._query_features(query_index, cache)
                best_positive = self._best_positive_index(
                    query_index, cache, query_features
                )
                soft_positives = self.dataset.soft_positives_per_query[query_index]
                neg_indexes = np.setdiff1d(
                    sampled_database, soft_positives, assume_unique=True
                )
                neg_indexes = self._hardest_negatives(
                    cache, query_features, neg_indexes
                )
                triplets.append((query_index, best_positive, *neg_indexes))
        return self._triplets_tensor(triplets)

    def _compute_cache(self, subset_ds, progress):
        collate_db, collate_q = _collate_fns(self.cfg.dataset)

        parent_ds, db_indices, q_indices = self._split_subset(subset_ds)
        subset_db = Subset(parent_ds, db_indices)
        subset_q = Subset(parent_ds, q_indices)

        db_sampler = (
            DistributedSampler(subset_db, shuffle=False, drop_last=False)
            if self.ddp_on
            else None
        )
        q_sampler = (
            DistributedSampler(subset_q, shuffle=False, drop_last=False)
            if self.ddp_on
            else None
        )
        cache_workers = _cache_num_workers(self.cfg)
        loader_kwargs = {
            "batch_size": self.cfg.infer_batch_size,
            "shuffle": False,
            "pin_memory": str(self.cfg.device).startswith("cuda"),
        }
        loader_kwargs.update(_worker_kwargs(self.cfg, cache_workers))
        db_loader = DataLoader(
            subset_db,
            sampler=db_sampler,
            collate_fn=collate_db,
            **loader_kwargs,
        )
        q_loader = DataLoader(
            subset_q,
            sampler=q_sampler,
            collate_fn=collate_q,
            **loader_kwargs,
        )
        cache = SparseFeatureCache((len(self.dataset), self.cfg.features_dim))

        with torch.no_grad(), _amp_context(self.cfg, self.device):
            self._run_descriptor_loader(
                db_loader,
                "db",
                dataloader_idx=0,
                cache=cache,
                progress=progress,
                disable=self.rank != 0,
            )
            self._run_descriptor_loader(
                q_loader,
                "q",
                dataloader_idx=1,
                cache=cache,
                progress=progress,
                disable=self.rank != 0,
            )
        return cache

    def _run_descriptor_loader(
        self,
        dataloader,
        mode,
        dataloader_idx,
        cache,
        progress,
        disable,
    ):
        local_idx_chunks = []
        local_feat_chunks = []
        iterable = _progress(dataloader, progress, f"cache/{mode}", disable=disable)
        for batch_idx, batch in enumerate(iterable):
            prediction = self.pl_module.predict_step(
                batch,
                batch_idx,
                dataloader_idx=dataloader_idx,
            )
            indexes = prediction["indices"]
            features = prediction["features"].float()
            local_idx_chunks.append(indexes.detach().cpu())
            local_feat_chunks.append(features.detach().cpu())

        if local_idx_chunks:
            local_idx_np = torch.cat(local_idx_chunks, dim=0).numpy()
            local_feat_np = torch.cat(local_feat_chunks, dim=0).numpy()
        else:
            local_idx_np = np.zeros(0, dtype=np.int64)
            local_feat_np = np.zeros((0, self.cfg.features_dim), dtype=np.float32)

        if self.ddp_on:
            gathered = [None] * self.world_size
            dist.all_gather_object(gathered, (local_idx_np, local_feat_np))
            for indexes_np, features_np in gathered:
                if len(indexes_np) > 0:
                    cache[indexes_np] = features_np
        elif len(local_idx_np) > 0:
            cache[local_idx_np] = local_feat_np

    def _split_subset(self, subset_ds):
        if isinstance(subset_ds, Subset):
            parent_ds = subset_ds.dataset
            subset_indices = [int(index) for index in subset_ds.indices]
        else:
            parent_ds = subset_ds
            subset_indices = list(range(len(parent_ds)))
        db_indices = [
            index for index in subset_indices if index < parent_ds.database_num
        ]
        q_indices = [
            index for index in subset_indices if index >= parent_ds.database_num
        ]
        return parent_ds, db_indices, q_indices

    def _sample_queries(self, weighted=False):
        if self.rank == 0:
            if weighted:
                sampled = np.random.choice(
                    self.dataset.queries_num,
                    self.cfg.cache_refresh_rate,
                    replace=False,
                    p=self.dataset.weights,
                )
            else:
                sampled = np.random.choice(
                    self.dataset.queries_num,
                    self.cfg.cache_refresh_rate,
                    replace=False,
                )
            sampled = sampled.astype(np.int64)
        else:
            sampled = np.zeros(self.cfg.cache_refresh_rate, dtype=np.int64)
        return self._broadcast_int_array(sampled)

    def _sample_database(self):
        if self.rank == 0:
            sampled = np.random.choice(
                self.dataset.database_num,
                self.dataset.neg_samples_num,
                replace=False,
            ).astype(np.int64)
        else:
            sampled = np.zeros(self.dataset.neg_samples_num, dtype=np.int64)
        return self._broadcast_int_array(sampled)

    def _broadcast_int_array(self, values):
        if self.world_size <= 1:
            return values.astype(np.int64, copy=False)

        tensor = torch.as_tensor(values, dtype=torch.long, device=self.device)
        tensor = self.trainer.strategy.broadcast(tensor, src=0)
        return tensor.detach().cpu().numpy().astype(np.int64, copy=False)

    def _positive_indexes(self, sampled_queries):
        positives = [
            self.dataset.hard_positives_per_query[int(index)]
            for index in sampled_queries
        ]
        positives = [int(item) for group in positives for item in group]
        return list(np.unique(positives))

    def _query_features(self, query_index, cache):
        features = cache[int(query_index) + self.dataset.database_num]
        if features is None:
            raise RuntimeError(
                f"Features were not computed for query index {int(query_index)}."
            )
        return features

    def _best_positive_index(self, query_index, cache, query_features):
        positives = self.dataset.hard_positives_per_query[int(query_index)]
        positive_features = cache[positives]
        index = faiss.IndexFlatL2(self.cfg.features_dim)
        index.add(positive_features)
        _, best_positive_num = index.search(query_features.reshape(1, -1), 1)
        return int(positives[int(best_positive_num[0, 0])])

    def _hardest_negatives(self, cache, query_features, neg_samples):
        neg_features = cache[neg_samples]
        index = faiss.IndexFlatL2(self.cfg.features_dim)
        index.add(neg_features)
        _, neg_nums = index.search(
            query_features.reshape(1, -1), self.dataset.negs_num_per_query
        )
        return neg_samples[neg_nums.reshape(-1)].astype(np.int32)

    def _triplets_tensor(self, triplets):
        if not triplets:
            return torch.zeros((0, 0), dtype=torch.long)
        return torch.as_tensor(triplets, dtype=torch.long)

    def _broadcast_triplets(self, triplets):
        if self.world_size <= 1:
            return triplets.cpu()

        if self.rank == 0:
            shape = torch.tensor(list(triplets.shape), dtype=torch.long, device=self.device)
            data = triplets.to(device=self.device, dtype=torch.long)
        else:
            shape = torch.zeros(2, dtype=torch.long, device=self.device)
            data = None

        shape = self.trainer.strategy.broadcast(shape, src=0)
        if self.rank != 0:
            data = torch.zeros(*shape.tolist(), dtype=torch.long, device=self.device)
        data = self.trainer.strategy.broadcast(data, src=0)
        try:
            self.trainer.strategy.barrier("triplet_cache_broadcast")
        except TypeError:
            self.trainer.strategy.barrier()
        return data.cpu()
