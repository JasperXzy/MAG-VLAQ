

# ======================== Single GPU ========================
python train.py --cuda 0  --dataset nuscenes  --camnames fl_f_fr_bl_b_br


python train.py --cuda 1  --dataset kitti360  --camnames 00


# ======================== Multi-GPU DDP ========================

torchrun --nproc_per_node=4 train.py --dataset kitti360 --camnames 00

# Example: 4 GPUs on a specific machine with custom settings
torchrun --nproc_per_node=4 train.py \
    --dataset kitti360 \
    --camnames 00 \
    --machine 5080 \
    --train_batch_size 4 \
    --infer_batch_size 16 \
    --num_workers 8

