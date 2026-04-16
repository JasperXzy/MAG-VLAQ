import torch


def _barrier(strategy, name):
    try:
        strategy.barrier(name)
    except TypeError:
        strategy.barrier()


def broadcast_triplets(triplets_ds, device, strategy, world_size, rank):
    if world_size <= 1:
        return

    if rank == 0:
        triplets = triplets_ds.triplets_global_indexes
        shape = torch.tensor(list(triplets.shape), dtype=torch.long, device=device)
    else:
        shape = torch.zeros(2, dtype=torch.long, device=device)

    shape = strategy.broadcast(shape, src=0)
    if rank == 0:
        data = triplets_ds.triplets_global_indexes.to(device=device, dtype=torch.long)
    else:
        data = torch.zeros(*shape.tolist(), dtype=torch.long, device=device)

    data = strategy.broadcast(data, src=0)
    triplets_ds.triplets_global_indexes = data.cpu()
    _barrier(strategy, "triplet_cache_broadcast")
