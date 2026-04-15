# ======================== Single GPU ========================
python train.py fit \
  --config configs/base.yaml \
  --config configs/kitti360.yaml \
  --config configs/exp/mm_dbvanilla2d.yaml \
  --cuda 0

python train.py fit \
  --config configs/base.yaml \
  --config configs/nuscenes.yaml \
  --config configs/exp/mm_dbvanilla2d.yaml \
  --cuda 1

# ======================== Multi-GPU DDP ========================
torchrun --nproc_per_node=4 train.py fit \
  --config configs/base.yaml \
  --config configs/kitti360.yaml \
  --config configs/exp/mm_dbvanilla2d.yaml \
  --trainer.strategy ddp_find_unused_parameters_true \
  --data.train_batch_size 4 \
  --data.infer_batch_size 16 \
  --data.num_workers 8
