
import os
import warnings
warnings.filterwarnings("ignore", message=".*xFormers is not available.*")
warnings.filterwarnings("ignore", message=".*xFormers is available.*")
warnings.filterwarnings("ignore", message=".*is_fx_tracing.*")
warnings.filterwarnings("ignore", message=r".*isinstance\(treespec, LeafSpec\).*")
warnings.filterwarnings("ignore", message=r".*Found .* module\(s\) in eval mode at the start of training.*")
warnings.filterwarnings("ignore", message=".*does not have many workers.*")
import logging
for _logger_name in (
    "httpx",
    "httpcore",
    "httpcore.http11",
    "httpcore.connection",
    "hpack",
    "torch.fx._symbolic_trace",
    "pytorch_lightning.utilities._pytree",
    "pytorch_lightning.loops.fit_loop",
    "pytorch_lightning.trainer.connectors.data_connector",
):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)
os.environ["OMP_NUM_THREADS"] = '16'

import argparse
import time


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--machine", type=str, default='5080')
    parser.add_argument("--dataset", type=str, default='kitti360') # kitti360  nuscenes  
    parser.add_argument("--datasets_folder", type=str, default='')
    parser.add_argument("--dataset_name", type=str, default='')
    parser.add_argument("--dataroot", type=str, 
                        )
    parser.add_argument("--maptype", type=str, default='satellite') # satellite  roadmap  terrain  hybrid
    parser.add_argument("--traindownsample", type=int, default=4) # 4
    parser.add_argument("--train_ratio", type=float, default=0.85) 
    # nuscenes: fl_f_fr_bl_b_br
    # kitti360: 00  0203 
    parser.add_argument("--camnames", type=str, default='00') 
    
    parser.add_argument("--train_batch_size", type=int, default=16, # 16
                        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images")
    parser.add_argument("--infer_batch_size", type=int, default=32, # 32
                        help="Batch size for inference (caching and testing)")
    
    parser.add_argument("--cache_refresh_rate", type=int, default=4000, # 1000, < len(train_dataset)
                        help="How often to refresh cache, in number of queries")
    parser.add_argument("--queries_per_epoch", type=int, default=16000, # 5000
                        help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25)
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10)
    parser.add_argument("--neg_samples_num", type=int, default=1000,
                        help="How many negatives to use to compute the hardest ones")
    parser.add_argument("--negs_num_per_query", type=int, default=10,
                        help="How many negatives to consider per each query in the loss")
    parser.add_argument("--epochs_num", type=int, default=100,
                        help="number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lrpc", type=float, default=1e-4)
    parser.add_argument("--lrdb", type=float, default=1e-5)
    parser.add_argument("--lrdino", type=float, default=0.0)
    parser.add_argument("--unfreeze_dino_mode", type=str, default="frozen", choices=["frozen", "last2", "full"])
    parser.add_argument("--dino_extract_blocks", type=str, default="7_15_23", help="Transformer block indices to extract multi-layer features from DINOv2")
    parser.add_argument("--utonia_pretrained", type=str, default="utonia", help="Load Utonia pretrained weights: 'none', 'utonia' (HF download), or local file path e.g. ~/.cache/utonia/ckpt/utonia.pth")
    parser.add_argument("--unfreeze_utonia_mode", type=str, default="last1", choices=["frozen", "last1", "full"], help="Utonia freeze mode: frozen=all frozen, last1=stage4 only (~40M/29%%), full=all (~137M)")
    parser.add_argument("--lrutonia", type=float, default=1e-5, help="Learning rate for unfrozen Utonia parameters")
    parser.add_argument("--utonia_extract_stages", type=str, default="0_2_4", help="PTv3 encoder stages to extract for ODE fusion")
    parser.add_argument("--amp_dtype", type=str, default="none", choices=["none", "bf16", "fp16"],
                        help="Mixed precision for forward/loss. 'bf16' is recommended on Ampere+ (4090). 'fp16' requires GradScaler (not wired up here).")
    parser.add_argument('--resize', type=int, default=[256,256], nargs=2, help="Resizing shape for images (HxW).") # database transform
    parser.add_argument('--color_jitter', type=float, default=0) # query transform
    parser.add_argument('--quant_size', type=float, default=2) # query transform
    parser.add_argument("--db_cropsize", type=int, default=256) # 256 384 512 640
    parser.add_argument("--db_resize", type=int, default=224)
    parser.add_argument("--db_jitter", type=float, default=0)
    parser.add_argument("--q_resize", type=int, default=224)
    parser.add_argument("--q_jitter", type=float, default=0)

    parser.add_argument("--share_db", type=str, default=False)
    parser.add_argument("--share_dbfe", type=str, default=False)
    parser.add_argument("--features_dim", type=int, default=256) 
    parser.add_argument("--read_pc", type=str, default=True) 

    # ==== for mm
    parser.add_argument('--mm_imgfe_dim', type=int, default=1024)
    parser.add_argument('--mm_voxfe_planes', type=str, default='64_128_256')  
    parser.add_argument('--mm_voxfe_dim', type=int, default=256)
    parser.add_argument('--mm_bevfe_planes', type=str, default='64_128_256')
    parser.add_argument('--mm_bevfe_dim', type=int, default=256)
    parser.add_argument('--mm_stg2fuse_dim', type=int, default=256) 
    parser.add_argument('--output_type', type=str, default='image_vox_shallow') # image_bev_shallow
    parser.add_argument('--output_l2', type=str, default=True)
    parser.add_argument('--final_type', type=str, default='imageorg_voxorg_shalloworg_stg2image_stg2vox') 
    parser.add_argument('--final_fusetype', type=str, default='add') # add  cat  catadd
    parser.add_argument('--final_l2', type=str, default=False)
    parser.add_argument('--image_embed', type=str, default='stg2image') # imageorg  stg2image
    parser.add_argument('--cloud_embed', type=str, default='stg2vox') # voxorg  stg2vox
    parser.add_argument('--image_weight', type=float, default=1)
    parser.add_argument('--image_learnweight', type=str, default=False)
    parser.add_argument('--vox_weight', type=float, default=1)
    parser.add_argument('--vox_learnweight', type=str, default=False)
    parser.add_argument('--shallow_weight', type=float, default=1)
    parser.add_argument('--shallow_learnweight', type=str, default=False)
    parser.add_argument('--diff_type', type=str, default='fcode@relu') # fcode@relu or fcode@sigmoid
    parser.add_argument('--diff_direction', type=str, default='backward') # forward  backward  
    parser.add_argument('--odeint_method', type=str, default='euler')
    parser.add_argument('--odeint_size', type=float, default=0.1)
    parser.add_argument('--tol', type=float, default=1e-3)
    parser.add_argument('--imagevoxorg_weight',  type=float, default=0) 
    parser.add_argument('--imagevoxorg_learnweight', type=str, default=False)
    parser.add_argument('--shalloworg_weight',   type=float, default=1.0) 
    parser.add_argument('--shalloworg_learnweight', type=str, default=False)
    parser.add_argument('--stg2imagevox_weight', type=float, default=0.1) 
    parser.add_argument('--stg2imagevox_learnweight', type=str, default=False)
    parser.add_argument('--stg2fuse_weight',     type=float, default=0)
    parser.add_argument('--stg2fuse_learnweight', type=str, default=False)
    parser.add_argument('--stg2nlayers', type=int, default=1)
    parser.add_argument('--stg2fuse_type', type=str, default='basic') # for stage 2
    parser.add_argument('--stg2_type', type=str, default='full') # for stage 2
    parser.add_argument('--stg2_useproj', type=str, default=True) # for stage 2
    parser.add_argument('--otherloss_type', type=str, default='bce') 
    parser.add_argument('--otherloss_weight', type=float, default=0.01)
    parser.add_argument('--tripletloss_weight', type=float, default=1)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--margin", type=float, default=0.1,
                        help="margin for the triplet loss")
    # partial  partial_sep
    parser.add_argument("--mining", type=str, default="partial_sep", choices=["partial", "full", "random", "msls_weighted"])

    # Paths parameters
    parser.add_argument("--save_dir", type=str, default="default",
                        help="Folder name of the current run (saved in ./logs/)")

    # Training parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str,help="Path to load checkpoint from, for resuming training or testing.",
                        default=None,
                        )


    # Other parameters
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
                        help="This includes pre/post-processing methods and prediction refinement")
    parser.add_argument("--majority_weight", type=float, default=0.01,
                        help="only for majority voting, scale factor, the higher it is the more importance is given to agreement")
    parser.add_argument("--efficient_ram_testing", action='store_true')
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 20], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    

    # Data augmentation parameters
    parser.add_argument("--brightness", type=float, default=0)
    parser.add_argument("--contrast", type=float, default=0)
    parser.add_argument("--saturation", type=float, default=0)
    parser.add_argument("--hue", type=float, default=0)
    parser.add_argument("--rand_perspective", type=float, default=0)
    parser.add_argument("--horizontal_flip", action='store_true')
    parser.add_argument("--random_resized_crop", type=float, default=0)
    parser.add_argument("--random_rotation", type=float, default=0)
    parser.add_argument('--exp_name', type=str, default='none')


    ##################################### parser
    args = parser.parse_args()
    opt_dict = vars(args)
    # print(args_dict)
    for k, v in opt_dict.items():
        if v in ['False','false']:
            opt_dict[k] = False
        elif v in ['True','true']:
            opt_dict[k] = True
        elif v in ['None','none']:
            opt_dict[k] = None
    args = argparse.Namespace(**opt_dict)

    if args.machine == '4090':
        if args.dataset == 'kitti360': 
            args.dataroot = '/mnt/sda/ZhengyiXu/datasets/cmvpr/kitti360/KITTI-360'
        elif args.dataset == 'nuscenes': 
            args.dataroot = '/mnt/sda/ZhengyiXu/datasets/radar/nuscenes'
        args.num_workers = 8
    elif args.machine == '5080':
        if args.dataset == 'kitti360': 
            args.dataroot = '/home/jasperxzy/Datasets/cmvpr/kitti360/KITTI-360'
        elif args.dataset == 'nuscenes': 
            args.dataroot = '/home/jasperxzy/Datasets/radar/nuscenes'
        args.num_workers = 8


    #####################################
    # When launched via torchrun (DDP), do NOT override CUDA_VISIBLE_DEVICES;
    # torchrun assigns GPUs through LOCAL_RANK.
    if 'RANK' not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda


    args.exp_name = ''
    args.exp_name += f'{args.seed}_'
    args.exp_name += f'ep{args.epochs_num}'
    args.exp_name += f'_{args.dataset}'
    args.exp_name += f'_{args.camnames}'
    args.exp_name += f'_{args.cache_refresh_rate}'
    args.exp_name += f'_{args.queries_per_epoch}'
    args.exp_name += f'_{args.maptype}'
    args.exp_name += f'_trbs{args.train_batch_size}'
    args.exp_name += f'_{args.infer_batch_size}'
    args.exp_name += f'_{args.traindownsample}'
    args.exp_name += f'_{args.train_ratio}'
    args.exp_name += f'_pc{args.read_pc}'




    #####################################
    args.output_type = args.output_type.split('_')
    args.final_type = args.final_type.split('_')




    #####################################


    if args.datasets_folder is None:
        try:
            args.datasets_folder = os.environ['DATASETS_FOLDER']
        except KeyError:
            raise Exception("You should set the parameter --datasets_folder or export " +
                            "the DATASETS_FOLDER environment variable as such \n" +
                            "export DATASETS_FOLDER=../datasets_vg/datasets")
    
    if args.queries_per_epoch % args.cache_refresh_rate != 0:
        raise ValueError("Ensure that queries_per_epoch is divisible by cache_refresh_rate, " +
                         f"because {args.queries_per_epoch} is not divisible by {args.cache_refresh_rate}")

    if args.mining == "msls_weighted" and args.dataset_name != "msls":
        raise ValueError("msls_weighted mining can only be applied to msls dataset, but you're using it on {args.dataset_name}")


    return args





def get_datetime():
    return time.strftime("%Y%m%d_%H%M")


def logging_info(args, message):
    with open(f'results/{args.exp_name}.txt', 'a') as f:
        f.write(message + '\n')
        # print(message)
    with open('results.txt', 'a') as f:
        f.write(message + '\n')



def logging_init(args):
    if not os.path.exists('results'):
        os.makedirs('results')
    with open(f'results/{args.exp_name}.txt', 'w') as f:
        f.write(get_datetime())
        f.write('\n')
        f.write(f'{args.exp_name}\n')
    with open('results.txt', 'w') as f:
        f.write(get_datetime())
        f.write('\n')
        f.write(f'{args.exp_name}\n')




def logging_end(args):
    with open(f'results/{args.exp_name}.txt', 'a') as f:
        f.write('\n')
        f.write(get_datetime())
    with open('results.txt', 'a') as f:
        f.write('\n')
        f.write(get_datetime())
