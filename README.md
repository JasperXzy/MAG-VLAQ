# Lightning Migration

This directory contains the isolated PyTorch Lightning migration. It does not
modify the original training code. After validation, copy it over the project
root with:

```bash
cp -r lightning/* .
```

Required runtime packages are the original project dependencies plus
`pytorch-lightning` and `PyYAML`.

Run a single-GPU experiment from the current isolated layout:

```bash
python lightning/train.py fit \
  --config lightning/configs/base.yaml \
  --config lightning/configs/kitti360.yaml \
  --config lightning/configs/exp/mm_dbvanilla2d.yaml \
  --cuda 0
```

Run DDP:

```bash
torchrun --nproc_per_node=4 lightning/train.py fit \
  --config lightning/configs/base.yaml \
  --config lightning/configs/kitti360.yaml \
  --config lightning/configs/exp/mm_dbvanilla2d.yaml \
  --trainer.strategy ddp_find_unused_parameters_true
```

Resume a Lightning checkpoint:

```bash
python lightning/train.py fit \
  --config lightning/configs/base.yaml \
  --config lightning/configs/kitti360.yaml \
  --config lightning/configs/exp/mm_dbvanilla2d.yaml \
  --ckpt_path logs/<exp>/checkpoints/last.ckpt
```

`epochs_num` and `queries_per_epoch/cache_refresh_rate` keep the original
semantics. The entrypoint derives Lightning `max_epochs` as
`epochs_num * ceil(queries_per_epoch / cache_refresh_rate)` and derives
`check_val_every_n_epoch` as the cache loop count unless these trainer values
are explicitly set in YAML or on the command line.

The entrypoint first converts YAML into the legacy `tools/options.py` argv
format before importing datasets and networks. This is required because the
existing business modules parse and cache options at import time.
