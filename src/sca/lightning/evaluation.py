import logging

import faiss
import numpy as np
import torch


def feature_store_length(test_ds, test_method):
    if test_method in {"nearest_crop", "maj_voting"}:
        return 5 * test_ds.queries_num + test_ds.database_num
    return len(test_ds)


def assign_features(all_features, indices, features, test_ds, test_method):
    indices = indices.detach().to(device=all_features.device, dtype=torch.long)
    features = features.to(device=all_features.device, dtype=all_features.dtype)

    if test_method in {"nearest_crop", "maj_voting"}:
        for row, index in enumerate(indices.tolist()):
            start = test_ds.database_num + (index - test_ds.database_num) * 5
            all_features[start:start + 5, :] = features[row * 5:(row + 1) * 5, :]
        return

    all_features[indices, :] = features


def compute_recall(cfg, queries_features, database_features, test_ds, test_method="hard_resize"):
    faiss_index = faiss.IndexFlatL2(cfg.features_dim)
    faiss_index.add(database_features)

    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_features, max(cfg.recall_values))

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
            top_n_voting("top1", predictions[q], distances[q], cfg.majority_weight)
            top_n_voting("top5", predictions[q], distances[q], cfg.majority_weight)
            top_n_voting("top10", predictions[q], distances[q], cfg.majority_weight)
            dists = distances[q].flatten()
            preds = predictions[q].flatten()
            sort_idx = np.argsort(dists)
            preds = preds[sort_idx]
            _, unique_idx = np.unique(preds, return_index=True)
            predictions[q, 0, :20] = preds[np.sort(unique_idx)][:20]
        predictions = predictions[:, 0, :20]

    positives_per_query = test_ds.get_positives()
    recalls = np.zeros(len(cfg.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(cfg.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    recalls = recalls / test_ds.queries_num * 100
    recalls_str = ", ".join(
        f"R@{val}: {rec:.1f}" for val, rec in zip(cfg.recall_values, recalls)
    )
    return recalls, recalls_str


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
