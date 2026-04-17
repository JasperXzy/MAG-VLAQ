def compute_triplet_loss(cfg, criterion_triplet, triplets_local_indexes, features):
    loss_triplet = features.new_tensor(0.0)
    local_batch = triplets_local_indexes.shape[0] // cfg.negs_num_per_query
    triplets_local_indexes = triplets_local_indexes.view(
        local_batch, cfg.negs_num_per_query, 3
    ).transpose(1, 0)

    for triplets in triplets_local_indexes:
        queries_indexes, positives_indexes, negatives_indexes = triplets.T
        loss_triplet = loss_triplet + criterion_triplet(
            features[queries_indexes],
            features[positives_indexes],
            features[negatives_indexes],
        )

    return loss_triplet / (local_batch * cfg.negs_num_per_query)
