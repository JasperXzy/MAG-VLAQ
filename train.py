
from tools import options as parser
from tools.options import logging_info, logging_init, get_datetime

import math
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
import os
from os.path import join
from datetime import datetime, timedelta
from torch.utils.data.dataloader import DataLoader
import time

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import util
import test
import commons
from tools.options import logging_info, logging_init, logging_end

from datasets.datasets_ws_kitti360 import KITTI360BaseDataset
from datasets.datasets_ws_kitti360 import KITTI360TripletsDataset
from datasets.datasets_ws_kitti360 import kitti360_collate_fn
from datasets.datasets_ws_kitti360 import kitti360_collate_fn_cache_db
from datasets.datasets_ws_kitti360 import kitti360_collate_fn_cache_q

from datasets.datasets_ws_nuscenes import NuScenesBaseDataset
from datasets.datasets_ws_nuscenes import NuScenesTripletsDataset
from datasets.datasets_ws_nuscenes import nuscenes_collate_fn
from datasets.datasets_ws_nuscenes import nuscenes_collate_fn_cache_db
from datasets.datasets_ws_nuscenes import nuscenes_collate_fn_cache_q



from network_mm.mm import MM    



from models_baseline.dbvanilla2d import DBVanilla2D

from compute_other_loss import compute_other_loss


# ======================== DDP Utilities ========================

def is_ddp():
    """Check if running in DDP mode (launched via torchrun)."""
    return dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_ddp() else 0

def get_world_size():
    return dist.get_world_size() if is_ddp() else 1

def is_main_process():
    return get_rank() == 0

def setup_ddp():
    """Initialize DDP process group. Must be called before any DDP operations."""
    # rank0 runs compute_triplets alone while others wait at barrier;
    # with Utonia's dense voxels this can exceed NCCL's default 600 s.
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=3600))
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def broadcast_triplets(triplets_ds, device):
    """Broadcast computed triplet indices from rank 0 to all ranks."""
    if not is_ddp():
        return

    rank = get_rank()
    # Broadcast the shape first so non-rank-0 can allocate
    if rank == 0:
        t = triplets_ds.triplets_global_indexes
        shape = torch.tensor(list(t.shape), dtype=torch.long, device=device)
    else:
        shape = torch.zeros(2, dtype=torch.long, device=device)
    dist.broadcast(shape, src=0)

    # Broadcast the actual data
    if rank == 0:
        data = triplets_ds.triplets_global_indexes.clone().to(device)
    else:
        data = torch.zeros(*shape.tolist(), dtype=torch.long, device=device)
    dist.broadcast(data, src=0)

    triplets_ds.triplets_global_indexes = data.cpu()


def get_amp_ctx(args):
    """Return a torch.autocast context manager based on --amp_dtype.
    'none' -> real no-op (nullcontext); 'bf16'/'fp16' -> cuda autocast.
    """
    import contextlib
    dt = getattr(args, 'amp_dtype', 'none')
    if dt == 'bf16':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    if dt == 'fp16':
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    return contextlib.nullcontext()


# ======================== Loss ========================

def compute_loss(args, criterion_triplet, triplets_local_indexes, features):
    loss_triplet = 0

    triplets_local_indexes = torch.transpose(
        triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
    for triplets in triplets_local_indexes:
        queries_indexes, positives_indexes, negatives_indexes = triplets.T
        loss_triplet += criterion_triplet(features[queries_indexes],
                                          features[positives_indexes],
                                          features[negatives_indexes])

    del features
    loss_triplet /= (args.train_batch_size * args.negs_num_per_query)

    return loss_triplet





def main():

    torch.backends.cudnn.benchmark = True  # Provides a speedup

    # ======================== DDP Setup ========================
    use_ddp = ('RANK' in os.environ and 'WORLD_SIZE' in os.environ)
    if use_ddp:
        local_rank = setup_ddp()
    else:
        local_rank = 0

    #### Initial setup: parser, logging...
    args = parser.parse_arguments()
    start_time = datetime.now()
    args.save_dir = join("logs", args.exp_name)

    # In DDP, override device to the assigned GPU
    if use_ddp:
        args.device = f'cuda:{local_rank}'

    # Only rank 0 handles logging setup and file I/O
    if is_main_process():
        logging_init(args)
        commons.setup_logging(args.save_dir)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.info(f"Arguments: {args}")
        logging.info(f"The outputs are being saved in {args.save_dir}")
        if use_ddp:
            logging.info(f"DDP enabled: world_size={get_world_size()}, using {get_world_size()} GPUs")
        else:
            logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")
    else:
        # Suppress logging on non-main ranks
        logging.basicConfig(level=logging.WARNING)

    # Seed: offset by rank so each GPU gets different augmentation
    commons.make_deterministic(args.seed + get_rank())

    #### Creation of Datasets
    if is_main_process():
        logging.debug(f"Loading dataset {args.dataset_name} from folder {args.datasets_folder}")

    if args.dataset == 'kitti360':
        triplets_ds = KITTI360TripletsDataset(args, args.datasets_folder, args.dataset_name, "train", args.negs_num_per_query)
        test_ds = KITTI360BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
    elif args.dataset == 'nuscenes':
        triplets_ds = NuScenesTripletsDataset(args, args.datasets_folder, args.dataset_name, "train", args.negs_num_per_query)
        test_ds = NuScenesBaseDataset(args, args.datasets_folder, args.dataset_name, "test")

    if is_main_process():
        logging.info(f"Train query set: {triplets_ds}")
        logging_info(args, f"Train query set: {triplets_ds}")
        logging.info(f"Test set: {test_ds}")
        logging_info(args, f"Test set: {test_ds}")


    #---- model db
    model = DBVanilla2D(mode='db', dim=args.features_dim, args=args)

    #---- model q
    modelq = MM(args=args)

    if is_main_process():
        num_params_q = sum(p.numel() for p in modelq.parameters())
        logging.info(f"Number of parameters in modelq: {num_params_q}")
        logging_info(args, f"Number of parameters in modelq: {num_params_q}")
        num_params_db = sum(p.numel() for p in model.parameters())
        logging.info(f"Number of parameters in model: {num_params_db}")
        logging_info(args, f"Number of parameters in model: {num_params_db}")

    # Move models to the correct device
    model = model.to(args.device)
    modelq = modelq.to(args.device)

    if is_main_process():
        logging_info(args, f"Model: {model.__class__.__name__}")
        logging.info(f"Model: {model.__class__.__name__}")
        logging_info(args, f"Modelq: {modelq.__class__.__name__}")
        logging.info(f"Modelq: {modelq.__class__.__name__}")
        logging_info(args, f"Device: {args.device}")
        logging.info(f"Device: {args.device}")


    # ======================== DDP: SyncBatchNorm + Wrap ========================
    if use_ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        modelq = nn.SyncBatchNorm.convert_sync_batchnorm(modelq)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        modelq = DDP(modelq, device_ids=[local_rank], find_unused_parameters=True)

    # Unwrapped references for inference, save/load, optimizer param groups
    model_without_ddp = model.module if use_ddp else model
    modelq_without_ddp = modelq.module if use_ddp else modelq


    # ==== model db optimizer params
    if isinstance(model_without_ddp, DBVanilla2D):
        params_db = []
        if getattr(args, 'lrdino', 0.0) > 0.0:
            dino_params = list(model_without_ddp.dbimage_fes.parameters())
            base_params = [p for n, p in model_without_ddp.named_parameters() if not n.startswith('dbimage_fes.')]

            dino_trainable = list(filter(lambda p: p.requires_grad, dino_params))
            if len(dino_trainable) > 0:
                params_db.append({'params': dino_trainable, 'lr': args.lrdino})

            base_trainable = list(filter(lambda p: p.requires_grad, base_params))
            if len(base_trainable) > 0:
                params_db.append({'params': base_trainable, 'lr': args.lrdb})
        else:
            params_db.append({'params': filter(lambda p: p.requires_grad, model_without_ddp.parameters()), 'lr': args.lrdb})


    # ==== model query optimizer params
    if isinstance(modelq_without_ddp, MM):
        params_q = []

        if getattr(args, 'lrdino', 0.0) > 0.0:
            dino_trainable = list(filter(lambda p: p.requires_grad, modelq_without_ddp.image_fe.parameters()))
            if len(dino_trainable) > 0:
                params_q.append({'params': dino_trainable, 'lr': args.lrdino})
        else:
            params_q.append({'params': filter(lambda p: p.requires_grad, modelq_without_ddp.image_fe.parameters()), 'lr': args.lr})

        params_q.append({'params': filter(lambda p: p.requires_grad, modelq_without_ddp.image_pool.parameters()), 'lr': args.lr})
        utonia_ptv3_trainable = [p for p in modelq_without_ddp.vox_fe.ptv3.parameters() if p.requires_grad]
        utonia_proj_trainable = [p for p in modelq_without_ddp.vox_fe.projs.parameters() if p.requires_grad]

        # Strict semantics: lrutonia=0 -> PTv3 NOT in optimizer (no fallback to lrpc).
        if getattr(args, 'lrutonia', 0.0) > 0.0 and len(utonia_ptv3_trainable) > 0:
            params_q.append({'params': utonia_ptv3_trainable, 'lr': args.lrutonia})

        if len(utonia_proj_trainable) > 0:
            params_q.append({'params': utonia_proj_trainable, 'lr': args.lrpc})

        # Any other trainable submodule inside vox_fe (not ptv3, not projs).
        other_vox = [p for n, p in modelq_without_ddp.vox_fe.named_parameters()
                     if p.requires_grad and not n.startswith('ptv3.') and not n.startswith('projs.')]
        if len(other_vox) > 0:
            params_q.append({'params': other_vox, 'lr': args.lrpc})
        params_q.append({'params': filter(lambda p: p.requires_grad, modelq_without_ddp.vox_pool.parameters()), 'lr': args.lrpc})
        params_q.append({'params': filter(lambda p: p.requires_grad, modelq_without_ddp.fuseblocktoshallow.parameters()), 'lr': args.lr})
        params_q.append({'params': filter(lambda p: p.requires_grad, modelq_without_ddp.stg2fuseblock.parameters()), 'lr': args.lr})
        params_q.append({'params': filter(lambda p: p.requires_grad, modelq_without_ddp.stg2fusefc.parameters()), 'lr': args.lr})

        # Projection layers
        params_q.append({'params': filter(lambda p: p.requires_grad, modelq_without_ddp.image_proj.parameters()), 'lr': args.lr})
        params_q.append({'params': filter(lambda p: p.requires_grad, modelq_without_ddp.vox_proj.parameters()), 'lr': args.lrpc})
        params_q.append({'params': filter(lambda p: p.requires_grad, modelq_without_ddp.stg2image_proj.parameters()), 'lr': args.lr})
        params_q.append({'params': filter(lambda p: p.requires_grad, modelq_without_ddp.stg2vox_proj.parameters()), 'lr': args.lrpc})

        params_q.append({'params': [modelq_without_ddp.image_weight], 'lr': args.lr})
        params_q.append({'params': [modelq_without_ddp.vox_weight], 'lr': args.lrpc})
        params_q.append({'params': [modelq_without_ddp.shallow_weight], 'lr': args.lr})
        params_q.append({'params': [modelq_without_ddp.imageorg_weight], 'lr': args.lr})
        params_q.append({'params': [modelq_without_ddp.voxorg_weight], 'lr': args.lr})
        params_q.append({'params': [modelq_without_ddp.shalloworg_weight], 'lr': args.lr})
        params_q.append({'params': [modelq_without_ddp.stg2image_weight], 'lr': args.lr})
        params_q.append({'params': [modelq_without_ddp.stg2vox_weight], 'lr': args.lr})
        params_q.append({'params': [modelq_without_ddp.stg2fuse_weight], 'lr': args.lr})


    optimizer = torch.optim.Adam(params_db)
    optimizerq = torch.optim.Adam(params_q)


    if is_main_process():
        num_params_db = sum(pp.numel() for p in optimizer.param_groups for pp in p['params'])
        num_params_q = sum(pp.numel() for p in optimizerq.param_groups for pp in p['params'])
        logging.info(f"Number of parameters in optimizerdb: {num_params_db}")
        logging_info(args, f"Number of parameters in optimizerdb: {num_params_db}")
        logging.info(f"Number of parameters in optimizerq: {num_params_q}")
        logging_info(args, f"Number of parameters in optimizerq: {num_params_q}")

        # Phase 0 instrumentation: per param_group lr + Utonia trainable ratio
        for i, g in enumerate(params_q):
            n = sum(p.numel() for p in g['params'])
            logging.info(f"[param_group {i}] {n/1e6:.3f}M params @ lr={g['lr']}")

        if hasattr(modelq_without_ddp, 'vox_fe') and hasattr(modelq_without_ddp.vox_fe, 'ptv3'):
            ptv3 = modelq_without_ddp.vox_fe.ptv3
            n_total = sum(p.numel() for p in ptv3.parameters())
            n_train = sum(p.numel() for p in ptv3.parameters() if p.requires_grad)
            logging.info(f"[PTv3] trainable {n_train/1e6:.2f}M / total {n_total/1e6:.2f}M")


    criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")


    #### Resume model, optimizer, and other training parameters
    if args.resume:
        model_without_ddp, modelq_without_ddp, optimizer, best_r5, start_epoch_num, not_improved_num = util.resume_train(
            args, model_without_ddp, modelq_without_ddp, optimizer)
        if is_main_process():
            logging.info(f"Resuming from epoch {start_epoch_num} with best recall@5 {best_r5:.1f}")
    else:
        best_r5 = start_epoch_num = not_improved_num = 0


    if is_main_process():
        logging.info(f"Output dimension of the model is {args.features_dim}")


    #### Training loop
    best_r1r5r10ep = [0, 0, 0, 0]
    for epoch_num in range(start_epoch_num, args.epochs_num):
        t0 = time.time()
        if is_main_process():
            logging.info(f"Start training epoch: {epoch_num:02d}")

        epoch_start_time = datetime.now()
        epoch_losses = np.zeros((0, 1), dtype=np.float32)

        # How many loops should an epoch last (default is 5000/1000=5)
        loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
        for loop_num in range(loops_num):
            if is_main_process():
                logging.debug(f"Cache: {loop_num} / {loops_num}")

            # ============ Compute triplets ============
            # All ranks participate in cache computation (data-parallel shard).
            # Rank 0 runs the CPU-bound mining loop, then broadcasts triplet
            # indices to the other ranks.
            if use_ddp:
                triplets_ds.is_inference = True
                if is_main_process():
                    logging.info('compute triplets')
                triplets_ds.compute_triplets(args, model_without_ddp, modelq_without_ddp)
                triplets_ds.is_inference = False
                dist.barrier()
                broadcast_triplets(triplets_ds, args.device)
            else:
                triplets_ds.is_inference = True
                logging.info('compute triplets')
                triplets_ds.compute_triplets(args, model, modelq)
                triplets_ds.is_inference = False

            # ============ Create training DataLoader ============
            triplets_ds.is_inference = False
            if args.dataset == 'kitti360':
                collate_fn = kitti360_collate_fn
            elif args.dataset == 'nuscenes':
                collate_fn = nuscenes_collate_fn

            if use_ddp:
                train_sampler = DistributedSampler(triplets_ds, shuffle=True)
                train_sampler.set_epoch(epoch_num * loops_num + loop_num)
                triplets_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                        batch_size=args.train_batch_size,
                                        collate_fn=collate_fn,
                                        pin_memory=True,
                                        drop_last=True,
                                        sampler=train_sampler)
            else:
                triplets_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                        batch_size=args.train_batch_size,
                                        collate_fn=collate_fn,
                                        pin_memory=(args.device == "cuda"),
                                        drop_last=True)

            model.train()
            modelq.train()


            if is_main_process():
                logging.info('start triplet training')

            for data_dict, triplets_local_indexes, _ in tqdm(
                triplets_dl,
                desc="train",
                disable=not is_main_process(),
                dynamic_ncols=True,
                leave=False,
                mininterval=1.0,
            ):

                for _k, _v in data_dict.items():
                    if isinstance(_v, torch.Tensor): data_dict[_k] = _v.to(args.device)


                with get_amp_ctx(args):
                    feats_ground = modelq(data_dict, mode='q') # [b,c]
                    feats_ground_embed = feats_ground['embedding']
                    feats_ground_embed = feats_ground_embed.unsqueeze(1) # [b,1,c]

                    # dbs
                    feats_aerial = model(data_dict, mode='db') # [b,11,c]
                    feats_aerial_embed = feats_aerial['embedding']

                    loss = 0
                    otherloss = compute_other_loss(feats_ground, feats_aerial, data_dict,
                                                   positive_thd=args.train_positives_dist_threshold,
                                                   negative_thd=args.val_positive_dist_threshold,
                                                   args=args)
                    loss += otherloss

                    # cat
                    feats = torch.cat((feats_ground_embed, feats_aerial_embed), dim=1)
                    feats = feats.view(-1, args.features_dim)
                    triplet_loss = compute_loss(args, criterion_triplet, triplets_local_indexes, feats)
                    loss += triplet_loss * args.tripletloss_weight
                    del feats

                optimizer.zero_grad()
                optimizerq.zero_grad()
                loss.backward()
                optimizer.step()
                optimizerq.step()

                # Keep track of all losses by appending them to epoch_losses
                batch_loss = loss.item()
                epoch_losses = np.append(epoch_losses, batch_loss)
                del loss


            if is_main_process() and len(epoch_losses) > 0:
                logging.debug(f"Epoch[{epoch_num:02d}]({loop_num}/{loops_num}): " +
                            f"current batch loss = {batch_loss:.4f}, " +
                            f"average epoch loss = {epoch_losses.mean():.4f}")



        if is_main_process():
            logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                        f"average epoch triplet loss = {epoch_losses.mean():.4f}")

        # ============ Test (rank 0 only) ============
        if is_main_process():
            recalls, _, _ = test.test(args, test_ds, model_without_ddp, modelq=modelq_without_ddp)

            is_best = sum(recalls[:3])>sum(best_r1r5r10ep[:3])
            if sum(recalls[:3])>sum(best_r1r5r10ep[:3]): # r@1 5 10
                best_r1r5r10ep[0] = recalls[0]
                best_r1r5r10ep[1] = recalls[1]
                best_r1r5r10ep[2] = recalls[2]
                best_r1r5r10ep[3] = epoch_num
            logging.info(f"Now : R@1 = {recalls[0]:.1f}   R@5 = {recalls[1]:.1f}   R@10 = {recalls[2]:.1f}   epoch = {epoch_num:d}")
            logging_info(args, f"Now : R@1 = {recalls[0]:.1f}   R@5 = {recalls[1]:.1f}   R@10 = {recalls[2]:.1f}   epoch = {epoch_num:d}")
            logging.info(f"Best: R@1 = {best_r1r5r10ep[0]:.1f}   R@5 = {best_r1r5r10ep[1]:.1f}   R@10 = {best_r1r5r10ep[2]:.1f}   epoch = {best_r1r5r10ep[3]:d}")
            logging_info(args, f"Best: R@1 = {best_r1r5r10ep[0]:.1f}   R@5 = {best_r1r5r10ep[1]:.1f}   R@10 = {best_r1r5r10ep[2]:.1f}   epoch = {best_r1r5r10ep[3]:d}")

            # Save checkpoint (always save unwrapped state_dict)
            util.save_checkpoint(args, {
                "epoch_num": epoch_num,
                'modelq_state_dict': modelq_without_ddp.state_dict(),
                "model_state_dict": model_without_ddp.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "recalls": recalls,
                "best_r5": best_r5,
                "not_improved_num": not_improved_num
            }, is_best, filename=f"ep@{epoch_num}__r1@{recalls[0]:.0f}.pth")

            logging_info(args, f'{get_datetime()}')
            logging.info(f'---------------------------------- epoch: {epoch_num}   time: {time.time()-t0:.2f}')
            logging_info(args, f'---------------------------------- epoch: {epoch_num}   time: {time.time()-t0:.2f}')

        # Synchronize all ranks before next epoch
        if use_ddp:
            dist.barrier()



    if is_main_process():
        logging_end(args)

        #### Test best model on test set
        best_model_path = join(args.save_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            best_model_state_dict = torch.load(best_model_path)["model_state_dict"]
            model_without_ddp.load_state_dict(best_model_state_dict)
            recalls, recalls_str = test.test(args, test_ds, model_without_ddp, test_method=args.test_method, modelq=modelq_without_ddp)
        else:
            logging.info("best_model.pth not found, skipping final test.")

    # Cleanup
    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
