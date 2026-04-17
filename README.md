# MAG-VLAQ

MAG-VLAQ is now packaged as a `src/mag_vlaq` Python project with a PyTorch
Lightning training entrypoint. The legacy isolated `lightning/` layout and
YAML-to-argparse bridge have been removed.

## Install

From the repository root:

```bash
pip install -e .
```

This installs the `mag_vlaq` package and the `mag-vlaq-train` console command. The root
`train.py` entrypoint also works without installation because it adds `src/` to
`sys.path`.

Utonia must be installed in the same Python environment before running the
query model:

```bash
pip install -e /path/to/Utonia
```

## Train

Single GPU:

```bash
python train.py fit \
  --config configs/base.yaml \
  --config configs/machines/5080.yaml \
  --config configs/exp/baseline.yaml
```

Installed entrypoint:

```bash
mag-vlaq-train fit \
  --config configs/base.yaml \
  --config configs/machines/5080.yaml \
  --config configs/exp/baseline.yaml
```

DDP:

```bash
torchrun --nproc_per_node=4 train.py fit \
  --config configs/base.yaml \
  --config configs/machines/4090.yaml \
  --config configs/exp/baseline.yaml \
  --trainer.strategy ddp_find_unused_parameters_true
```

Resume:

```bash
python train.py fit \
  --config configs/base.yaml \
  --config configs/machines/5080.yaml \
  --config configs/exp/baseline.yaml
  --ckpt_path logs/<exp>/checkpoints/last.ckpt
```

Small smoke run:

```bash
python train.py fit \
  --config configs/base.yaml \
  --config configs/machines/5080.yaml \
  --config configs/exp/baseline.yaml \
  --data.epochs_num 1 \
  --data.queries_per_epoch 400 \
  --data.cache_refresh_rate 400 \
  --data.neg_samples_num 400 \
  --data.infer_batch_size 4 \
  --data.num_workers 4 \
  --data.cache_num_workers 0 \
  --trainer.devices 1
```

`epochs_num` and `queries_per_epoch/cache_refresh_rate` keep the original
semantics. The entrypoint derives Lightning `max_epochs` as
`epochs_num * ceil(queries_per_epoch / cache_refresh_rate)` and sets validation
to run once per original epoch unless the trainer config explicitly overrides
those values.

## Logging

Metrics are logged through Lightning loggers:

- CSVLogger is enabled by default and writes under `save_dir/lightning_logs`.
- LitLogger is optional. Enable it with `--logging.litlogger.enabled true` or in
  YAML.
- Runtime logs still go to `info.log` and `debug.log` under `save_dir`.

The old `results.txt` and `results/{exp_name}.txt` outputs have been removed.

## Layout

```text
MAG-VLAQ/
├── pyproject.toml
├── train.py
├── configs/
├── src/mag_vlaq/
│   ├── config.py
│   ├── data/
│   ├── losses/
│   ├── models/
│   └── lightning/
└── docs/
```

Important modules:

- `src/mag_vlaq/config.py`: structured config loaded from layered YAML and CLI
  overrides.
- `src/mag_vlaq/lightning/module.py`: `MagVlaqModule` training, validation, and predict
  steps.
- `src/mag_vlaq/lightning/datamodule.py`: train and validation dataloaders.
- `src/mag_vlaq/lightning/triplet_cache.py`: triplet cache refresh and mining.
- `src/mag_vlaq/lightning/optim.py`: single Adam optimizer with DB/query param
  groups.

## Checks

Useful local checks after refactors:

```bash
python -m compileall src train.py
PYTHONPATH=src python -c "import mag_vlaq.lightning.cli"
python -m pip install -e . --no-deps --dry-run --no-build-isolation
```
