import math
import sys

import torch

try:
    import pytorch_lightning as pl
except ImportError as exc:  # pragma: no cover - depends on environment
    raise ImportError(
        "PyTorch Lightning is required for the Lightning training entrypoint."
    ) from exc

from sca.config import Config
from sca.lightning.runtime import (
    build_loggers,
    build_trainer_kwargs,
    import_symbol,
    initialise_litlogger_capture,
    log_startup,
    rank,
    setup_runtime_logging,
    suppress_noisy_runtime_warnings,
)


class SCALightningCLI:
    """Lightning training entrypoint backed by the structured SCA Config."""

    def __init__(self, model_class: str, datamodule_class: str):
        suppress_noisy_runtime_warnings()
        cfg, command = Config.from_argv(sys.argv[1:])
        if command not in {"fit"}:
            raise ValueError(f"Only the 'fit'/'train' command is implemented, got: {command}")

        cfg.apply_runtime_environment()
        loops_num = math.ceil(cfg.queries_per_epoch / cfg.cache_refresh_rate)
        logger_config = build_loggers(cfg)
        initialise_litlogger_capture(logger_config, cfg)
        trainer_config = build_trainer_kwargs(cfg, loops_num, logger_config)

        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
        pl.seed_everything(cfg.seed + rank(), workers=True)
        setup_runtime_logging(cfg)

        model = import_symbol(model_class)(cfg)
        datamodule = import_symbol(datamodule_class)(cfg)
        trainer = pl.Trainer(**trainer_config)
        log_startup(cfg, trainer_config, loops_num)
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)


def main():
    SCALightningCLI(
        model_class="sca.lightning.module:SCAModule",
        datamodule_class="sca.lightning.datamodule:SCADataModule",
    )
