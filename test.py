import contextlib
import logging

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

from datasets.datasets_ws_kitti360 import (
    kitti360_collate_fn_cache_db,
    kitti360_collate_fn_cache_q,
)
from datasets.datasets_ws_nuscenes import (
    nuscenes_collate_fn_cache_db,
    nuscenes_collate_fn_cache_q,
)

_TEST_PROGRESS_CALLBACK = None


def set_progress_callback(callback):
    global _TEST_PROGRESS_CALLBACK
    _TEST_PROGRESS_CALLBACK = callback


def _progress(iterable, args=None, desc=None, disable=False):
    if _TEST_PROGRESS_CALLBACK is not None and not disable:
        total = len(iterable) if hasattr(iterable, "__len__") else None
        _TEST_PROGRESS_CALLBACK("start", desc, total)
        try:
            for item in iterable:
                yield item
                _TEST_PROGRESS_CALLBACK("advance", desc, 1)
        finally:
            _TEST_PROGRESS_CALLBACK("close", desc, None)
        return

    disable = disable or bool(getattr(args, "disable_dataset_tqdm", False))
    yield from tqdm(
        iterable,
        desc=f"eval/{desc}" if desc else "eval",
        disable=disable,
        dynamic_ncols=True,
        leave=False,
        mininterval=1.0,
    )


def _pin_memory(args):
    return str(getattr(args, "device", "")).startswith("cuda")


def _collate_fns(args):
    if args.dataset == "kitti360":
        return kitti360_collate_fn_cache_db, kitti360_collate_fn_cache_q
    if args.dataset == "nuscenes":
        return nuscenes_collate_fn_cache_db, nuscenes_collate_fn_cache_q
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def compute_recall(args, queries_features, database_features, test_ds, test_method="hard_resize"):
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)

    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_features, max(args.recall_values))

    if test_method == "nearest_crop":
        distances = np.reshape(distances, (test_ds.queries_num, 20 * 5))
        predictions = np.reshape(predictions, (test_ds.queries_num, 20 * 5))
        for q in range(test_ds.queries_num):
            sort_idx = np.argsort(distances[q])
            predictions[q] = predictions[q, sort_idx]
            _, unique_idx = np.unique(predictions[q], return_index=True)
            predictions[q, :20] = predictions[q, np.sort(unique_idx)][:20]
        predictions = predictions[:, :20]
    elif test_method == "maj_voting":
        distances = np.reshape(distances, (test_ds.queries_num, 5, 20))
        predictions = np.reshape(predictions, (test_ds.queries_num, 5, 20))
        for q in range(test_ds.queries_num):
            top_n_voting("top1", predictions[q], distances[q], args.majority_weight)
            top_n_voting("top5", predictions[q], distances[q], args.majority_weight)
            top_n_voting("top10", predictions[q], distances[q], args.majority_weight)
            dists = distances[q].flatten()
            preds = predictions[q].flatten()
            sort_idx = np.argsort(dists)
            preds = preds[sort_idx]
            _, unique_idx = np.unique(preds, return_index=True)
            predictions[q, 0, :20] = preds[np.sort(unique_idx)][:20]
        predictions = predictions[:, 0, :20]

    positives_per_query = test_ds.get_positives()
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    recalls = recalls / test_ds.queries_num * 100
    recalls_str = ", ".join(
        f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)
    )
    return recalls, recalls_str


def test(args, test_ds, model, test_method="hard_resize", pca=None, modelq=None):
    assert test_method in [
        "hard_resize",
        "single_query",
        "central_crop",
        "five_crops",
        "nearest_crop",
        "maj_voting",
    ], f"test_method can't be {test_method}"

    model = model.eval()
    modelq = modelq.eval() if modelq is not None else model
    collate_db, collate_q = _collate_fns(args)

    amp_dtype = getattr(args, "amp_dtype", "none")
    if amp_dtype == "bf16":
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif amp_dtype == "fp16":
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        amp_ctx = contextlib.nullcontext()

    with torch.no_grad(), amp_ctx:
        logging.debug("Extracting database features for evaluation/testing")
        test_ds.test_method = "hard_resize"
        database_subset_ds = Subset(test_ds, list(range(test_ds.database_num)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds,
            num_workers=args.num_workers,
            batch_size=args.infer_batch_size,
            pin_memory=_pin_memory(args),
            collate_fn=collate_db,
        )

        if test_method in {"nearest_crop", "maj_voting"}:
            all_features = np.empty(
                (5 * test_ds.queries_num + test_ds.database_num, args.features_dim),
                dtype="float32",
            )
        else:
            all_features = np.empty((len(test_ds), args.features_dim), dtype="float32")

        db_locations = []
        for data_dict, indices in _progress(database_dataloader, args=args, desc="db"):
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.to(args.device)
            features = model(data_dict, mode="db")["embedding"]
            features = features.float().cpu().numpy()
            if pca is not None:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features
            if args.dataset == "nuscenes":
                db_locations.extend(data_dict["db_location"])

        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
        test_ds.test_method = test_method
        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.database_num, test_ds.database_num + test_ds.queries_num))
        )
        queries_dataloader = DataLoader(
            dataset=queries_subset_ds,
            num_workers=args.num_workers,
            batch_size=queries_infer_batch_size,
            pin_memory=_pin_memory(args),
            collate_fn=collate_q,
        )

        q_locations = []
        for data_dict, indices in _progress(queries_dataloader, args=args, desc="q"):
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.to(args.device)
            features = modelq(data_dict, mode="q")["embedding"]
            if test_method == "five_crops":
                features = torch.stack(torch.split(features, 5)).mean(1)
            features = features.float().cpu().numpy()
            if pca is not None:
                features = pca.transform(features)
            if test_method in {"nearest_crop", "maj_voting"}:
                start_idx = test_ds.database_num + (indices[0] - test_ds.database_num) * 5
                end_idx = start_idx + indices.shape[0] * 5
                all_features[np.arange(start_idx, end_idx), :] = features
            else:
                all_features[indices.numpy(), :] = features
            if args.dataset == "nuscenes":
                q_locations.extend(data_dict["query_location"])

    queries_features = all_features[test_ds.database_num :]
    database_features = all_features[: test_ds.database_num]
    if args.dataset == "nuscenes":
        assert len(q_locations) == len(queries_features)
        assert len(db_locations) == len(database_features)

    recalls, recalls_str = compute_recall(
        args, queries_features, database_features, test_ds, test_method
    )
    return recalls, recalls_str, None


def top_n_voting(topn, predictions, distances, maj_weight):
    if topn == "top1":
        n = 1
        selected = 0
    elif topn == "top5":
        n = 5
        selected = slice(0, 5)
    elif topn == "top10":
        n = 10
        selected = slice(0, 10)
    else:
        raise ValueError(topn)

    vals, counts = np.unique(predictions[:, selected], return_counts=True)
    for val, count in zip(vals[counts > 1], counts[counts > 1]):
        mask = predictions[:, selected] == val
        distances[:, selected][mask] -= maj_weight * count / n
