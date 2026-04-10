
from tools import options as parser
opt = parser.parse_arguments()
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
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
import time

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import util
import test
import commons
from model.functional import sare_ind, sare_joint
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
import torchvision.models as TVM

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
    dist.init_process_group(backend='nccl')
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


# ======================== Loss ========================

def compute_loss(args, criterion_triplet, triplets_local_indexes, features):
    loss_triplet = 0

    if args.criterion == "triplet":
        triplets_local_indexes = torch.transpose(
            triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
        for triplets in triplets_local_indexes:
            queries_indexes, positives_indexes, negatives_indexes = triplets.T
            loss_triplet += criterion_triplet(features[queries_indexes],
                                            features[positives_indexes],
                                            features[negatives_indexes])
    elif args.criterion == 'sare_joint':
        # sare_joint needs to receive all the negatives at once
        triplet_index_batch = triplets_local_indexes.view(args.train_batch_size, 10, 3)
        for batch_triplet_index in triplet_index_batch:
            q = features[batch_triplet_index[0, 0]].unsqueeze(0)  # obtain query as tensor of shape 1xn_features
            p = features[batch_triplet_index[0, 1]].unsqueeze(0)  # obtain positive as tensor of shape 1xn_features
            n = features[batch_triplet_index[:, 2]]               # obtain negatives as tensor of shape 10xn_features
            loss_triplet += criterion_triplet(q, p, n)
    elif args.criterion == "sare_ind":
        for triplet in triplets_local_indexes:
            # triplet is a 1-D tensor with the 3 scalars indexes of the triplet
            q_i, p_i, n_i = triplet
            loss_triplet += criterion_triplet(features[q_i:q_i+1], features[p_i:p_i+1], features[n_i:n_i+1])

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
        if use_ddp:
            logging_init()
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

    if opt.dataset == 'kitti360':
        triplets_ds = KITTI360TripletsDataset(args, args.datasets_folder, args.dataset_name, "train", args.negs_num_per_query)
        test_ds = KITTI360BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
    elif opt.dataset == 'nuscenes':
        triplets_ds = NuScenesTripletsDataset(args, args.datasets_folder, args.dataset_name, "train", args.negs_num_per_query)
        test_ds = NuScenesBaseDataset(args, args.datasets_folder, args.dataset_name, "test")

    if is_main_process():
        logging.info(f"Train query set: {triplets_ds}")
        logging_info(f"Train query set: {triplets_ds}")
        logging.info(f"Test set: {test_ds}")
        logging_info(f"Test set: {test_ds}")


    #---- model db
    if args.modeldb == 'vanilla2d':
        model = DBVanilla2D(mode='db', dim=args.features_dim)

    #---- model q
    if args.modelq == 'mm':
        modelq = MM()
    else:
        raise NotImplementedError

    if is_main_process():
        num_params_q = sum(p.numel() for p in modelq.parameters())
        logging.info(f"Number of parameters in modelq: {num_params_q}")
        logging_info(f"Number of parameters in modelq: {num_params_q}")
        num_params_db = sum(p.numel() for p in model.parameters())
        logging.info(f"Number of parameters in model: {num_params_db}")
        logging_info(f"Number of parameters in model: {num_params_db}")

    # Move models to the correct device
    model = model.to(args.device)
    modelq = modelq.to(args.device)

    if is_main_process():
        logging_info(f"Model: {model.__class__.__name__}")
        logging.info(f"Model: {model.__class__.__name__}")
        logging_info(f"Modelq: {modelq.__class__.__name__}")
        logging.info(f"Modelq: {modelq.__class__.__name__}")
        logging_info(f"Device: {args.device}")
        logging.info(f"Device: {args.device}")


    if args.aggregation in ["netvlad", "crn"]:  # If using NetVLAD layer, initialize it
        if not args.resume:
            triplets_ds.is_inference = True
            model.aggregation.initialize_netvlad_layer(args, triplets_ds, model.backbone)
            modelq.aggregation.initialize_netvlad_layer(args, triplets_ds, modelq.backbone)
        args.features_dim *= args.netvlad_clusters


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
        if getattr(args, 'lrdino', 0.0) > 0.0 and args.dbimage_fe == 'dinov2_vitl14':
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

        if getattr(args, 'lrdino', 0.0) > 0.0 and args.mm_imgfe == 'dinov2_vitl14':
            dino_trainable = list(filter(lambda p: p.requires_grad, modelq_without_ddp.image_fe.parameters()))
            if len(dino_trainable) > 0:
                params_q.append({'params': dino_trainable, 'lr': args.lrdino})
        else:
            params_q.append({'params': filter(lambda p: p.requires_grad, modelq_without_ddp.image_fe.parameters()), 'lr': args.lr})

        params_q.append({'params': filter(lambda p: p.requires_grad, modelq_without_ddp.image_pool.parameters()), 'lr': args.lr})
        if getattr(args, 'lrutonia', 0.0) > 0.0 and getattr(args, 'mm_voxfe_arch', 'minkfpn') == 'utonia':
            utonia_ptv3_trainable = list(filter(lambda p: p.requires_grad, modelq_without_ddp.vox_fe.ptv3.parameters()))
            utonia_proj_trainable = list(filter(lambda p: p.requires_grad, modelq_without_ddp.vox_fe.projs.parameters()))
            if len(utonia_ptv3_trainable) > 0:
                params_q.append({'params': utonia_ptv3_trainable, 'lr': args.lrutonia})
            if len(utonia_proj_trainable) > 0:
                params_q.append({'params': utonia_proj_trainable, 'lr': args.lrpc})
        else:
            params_q.append({'params': filter(lambda p: p.requires_grad, modelq_without_ddp.vox_fe.parameters()), 'lr': args.lrpc})
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


    if opt.share_qdb == True:
        model = modelq
        model_without_ddp = modelq_without_ddp
        params_db = [{'params': torch.empty(0), 'lr': args.lrdb}]
        if is_main_process():
            logging.info(f"Sharing weights... {modelq_without_ddp.__class__.__name__} and {model_without_ddp.__class__.__name__}")



    if args.aggregation == "crn":
        crn_params = list(model_without_ddp.aggregation.crn.parameters())
        net_params = list(model_without_ddp.backbone.parameters()) + \
            list([m[1] for m in model_without_ddp.aggregation.named_parameters() if not m[0].startswith('crn')])
        if args.optim == "adam":
            optimizer = torch.optim.Adam([{'params': crn_params, 'lr': args.lr_crn_layer},
                                        {'params': net_params, 'lr': args.lr_crn_net}])
            if is_main_process():
                logging.info("You're using CRN with Adam, it is advised to use SGD")
        elif args.optim == "sgd":
            optimizer = torch.optim.SGD([{'params': crn_params, 'lr': args.lr_crn_layer, 'momentum': 0.9, 'weight_decay': 0.001},
                                        {'params': net_params, 'lr': args.lr_crn_net, 'momentum': 0.9, 'weight_decay': 0.001}])
    else:
        if args.optim == "adam":
            optimizer = torch.optim.Adam(params_db)
            optimizerq = torch.optim.Adam(params_q)


    if is_main_process():
        num_params_db = sum(pp.numel() for p in optimizer.param_groups for pp in p['params'])
        num_params_q = sum(pp.numel() for p in optimizerq.param_groups for pp in p['params'])
        logging.info(f"Number of parameters in optimizerdb: {num_params_db}")
        logging_info(f"Number of parameters in optimizerdb: {num_params_db}")
        logging.info(f"Number of parameters in optimizerq: {num_params_q}")
        logging_info(f"Number of parameters in optimizerq: {num_params_q}")


    if args.criterion == "triplet":
        criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")
    elif args.criterion == "sare_ind":
        criterion_triplet = sare_ind
    elif args.criterion == "sare_joint":
        criterion_triplet = sare_joint


    #### Resume model, optimizer, and other training parameters
    if args.resume:
        if args.aggregation != 'crn':
            model_without_ddp, modelq_without_ddp, optimizer, best_r5, start_epoch_num, not_improved_num = util.resume_train(args, model_without_ddp, modelq_without_ddp, optimizer)
        else:
            model_without_ddp, modelq_without_ddp, _, best_r5, start_epoch_num, not_improved_num = util.resume_train(args, model_without_ddp, modelq_without_ddp, strict=False)
        if is_main_process():
            logging.info(f"Resuming from epoch {start_epoch_num} with best recall@5 {best_r5:.1f}")
    else:
        best_r5 = start_epoch_num = not_improved_num = 0


    if is_main_process():
        if args.backbone.startswith('vit'):
            logging.info(f"Output dimension of the model is {args.features_dim}")
        else:
            logging.info(f"Output dimension of the model is {args.features_dim}, with {util.get_flops(model_without_ddp, args.resize)}")


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
            # Rank 0 computes triplets (inference), then broadcasts indices to all ranks.
            if use_ddp:
                if is_main_process():
                    triplets_ds.is_inference = True
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
            if opt.dataset == 'kitti360':
                collate_fn = kitti360_collate_fn
            elif opt.dataset == 'nuscenes':
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

            for data_dict, triplets_local_indexes, _ in tqdm(triplets_dl, disable=not is_main_process()):

                for _k, _v in data_dict.items():
                    if isinstance(_v, torch.Tensor): data_dict[_k] = _v.to(args.device)


                with torch.set_grad_enabled(args.train_modelq):
                    feats_ground = modelq(data_dict, mode='q') # [b,c]
                    feats_ground_embed = feats_ground['embedding']
                    feats_ground_embed = feats_ground_embed.unsqueeze(1) # [b,1,c]



                # dbs
                with torch.set_grad_enabled(args.train_modeldb):
                    feats_aerial = model(data_dict, mode='db') # [b,11,c]
                    feats_aerial_embed = feats_aerial['embedding']


                loss = 0
                if opt.modelq == 'mm':
                    otherloss = compute_other_loss(feats_ground, feats_aerial, data_dict,
                                                positive_thd=opt.train_positives_dist_threshold,
                                                negative_thd=opt.val_positive_dist_threshold)
                    loss += otherloss



                # cat
                feats = torch.cat((feats_ground_embed, feats_aerial_embed), dim=1)
                feats = feats.view(-1, args.features_dim)
                triplet_loss = compute_loss(args, criterion_triplet, triplets_local_indexes, feats)
                loss += triplet_loss * opt.tripletloss_weight
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
            logging_info(f"Now : R@1 = {recalls[0]:.1f}   R@5 = {recalls[1]:.1f}   R@10 = {recalls[2]:.1f}   epoch = {epoch_num:d}")
            logging.info(f"Best: R@1 = {best_r1r5r10ep[0]:.1f}   R@5 = {best_r1r5r10ep[1]:.1f}   R@10 = {best_r1r5r10ep[2]:.1f}   epoch = {best_r1r5r10ep[3]:d}")
            logging_info(f"Best: R@1 = {best_r1r5r10ep[0]:.1f}   R@5 = {best_r1r5r10ep[1]:.1f}   R@10 = {best_r1r5r10ep[2]:.1f}   epoch = {best_r1r5r10ep[3]:d}")

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

            logging_info(f'{get_datetime()}')
            logging.info(f'---------------------------------- epoch: {epoch_num}   time: {time.time()-t0:.2f}')
            logging_info(f'---------------------------------- epoch: {epoch_num}   time: {time.time()-t0:.2f}')

        # Synchronize all ranks before next epoch
        if use_ddp:
            dist.barrier()



    if is_main_process():
        logging_end()

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
    # logging_init is called inside main() after DDP setup, guarded by is_main_process().
    # We call it here only for the non-DDP case (before main sets up DDP).
    use_ddp = ('RANK' in os.environ and 'WORLD_SIZE' in os.environ)
    if not use_ddp:
        logging_init()
    main()
    # logging_end is called inside main() for rank 0.
    if not use_ddp:
        logging_end()
