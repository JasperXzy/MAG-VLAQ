from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from datetime import timedelta
from typing import Any, Dict
import warnings

import torch

try:
    import pytorch_lightning as pl
except ImportError as exc:  # pragma: no cover - depends on environment
    raise ImportError(
        "PyTorch Lightning is required for the Lightning training entrypoint."
    ) from exc

try:
    from pytorch_lightning.strategies import DDPStrategy
except Exception:  # pragma: no cover - older Lightning versions
    DDPStrategy = None

from sca.config import Config
from sca.lightning.logging_utils import setup_logging


def suppress_noisy_runtime_warnings() -> None:
    for pattern in (
        r".*xFormers is not available.*",
        r".*is_fx_tracing.*",
        r".*isinstance\(treespec, LeafSpec\).*",
        r".*Found .* module\(s\) in eval mode at the start of training.*",
        r".*val_dataloader.*does not have many workers.*",
        r".*tensorboardX.*has been removed as a dependency.*",
    ):
        warnings.filterwarnings("ignore", message=pattern)

    for logger_name in (
        "faiss",
        "faiss.loader",
        "fsspec",
        "fsspec.local",
        "matplotlib",
        "matplotlib.font_manager",
        "PIL",
        "torch.fx._symbolic_trace",
        "pytorch_lightning.utilities._pytree",
        "pytorch_lightning.loops.fit_loop",
        "pytorch_lightning.trainer.connectors.data_connector",
        "pytorch_lightning.trainer.connectors.logger_connector.logger_connector",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def import_symbol(target: str):
    module_name, symbol = target.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol)


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_rank_zero() -> bool:
    return rank() == 0


def build_loggers(cfg: Config):
    csv_config = cfg.logging.csv or {}
    lit_config = cfg.logging.litlogger or {}
    loggers = []

    if csv_config.get("enabled", True):
        loggers.append(
            pl.loggers.CSVLogger(
                save_dir=csv_config.get("save_dir") or cfg.save_dir,
                name=csv_config.get("name") or "lightning_logs",
            )
        )

    if lit_config.get("enabled", False):
        LitLogger = _get_litlogger_class()
        if LitLogger is None or importlib.util.find_spec("litlogger") is None:
            raise ImportError(
                "LitLogger was enabled, but it is not available. "
                "Install it with `pip install litlogger` or `pip install lightning[extra]`."
            )
        lit_kwargs = {
            "root_dir": lit_config.get("root_dir") or cfg.save_dir,
            "name": lit_config.get("name") or cfg.exp_name,
            "teamspace": lit_config.get("teamspace"),
            "metadata": _metadata_to_strings(lit_config.get("metadata", {})),
            "store_step": lit_config.get("store_step", True),
            "log_model": lit_config.get("log_model", False),
            "save_logs": lit_config.get("save_logs", True),
            "checkpoint_name": lit_config.get("checkpoint_name"),
        }
        lit_kwargs = {key: value for key, value in lit_kwargs.items() if value is not None}
        loggers.append(_disable_litlogger_graph_logging(LitLogger(**lit_kwargs)))

    if not loggers:
        return False
    return loggers[0] if len(loggers) == 1 else loggers


def initialise_litlogger_capture(logger_config: Any, cfg: Config) -> None:
    lit_config = cfg.logging.litlogger or {}
    if (
        not is_rank_zero()
        or not bool(lit_config.get("enabled", False) and lit_config.get("save_logs", True))
        or os.environ.get("_IN_PTY_RECORDER") == "1"
    ):
        return
    for logger in _iter_loggers(logger_config):
        if type(logger).__name__ == "LitLogger":
            _ = logger.experiment
            return


def build_trainer_kwargs(cfg: Config, loops_num: int, logger_config: Any) -> Dict[str, Any]:
    trainer_config = cfg.trainer.to_kwargs()
    cuda_available = torch.cuda.is_available()
    trainer_config.setdefault("accelerator", "gpu" if cuda_available else "auto")
    trainer_config.setdefault("devices", "auto")
    trainer_config.setdefault("precision", cfg.default_precision())
    trainer_config.setdefault("reload_dataloaders_every_n_epochs", 1)
    trainer_config.setdefault("check_val_every_n_epoch", loops_num)
    trainer_config.setdefault("max_epochs", cfg.epochs_num * loops_num)
    trainer_config.setdefault("default_root_dir", cfg.save_dir)
    trainer_config.setdefault("num_sanity_val_steps", 0)
    trainer_config.setdefault("log_every_n_steps", 20)
    trainer_config.setdefault("enable_model_summary", False)
    trainer_config.setdefault("sync_batchnorm", cuda_available)
    trainer_config["strategy"] = _normalise_strategy(
        trainer_config.get("strategy", "auto")
    )

    callbacks = trainer_config.pop("callbacks", None) or []
    has_progress_bar = any(
        type(callback).__name__.endswith("ProgressBar") for callback in callbacks
    )
    from sca.lightning.callbacks import TripletCacheRefreshCallback

    callbacks.append(TripletCacheRefreshCallback(loops_num=loops_num))
    if trainer_config.get("enable_progress_bar", True) and not has_progress_bar:
        callbacks.append(
            pl.callbacks.RichProgressBar(leave=True, console_kwargs={"stderr": True})
        )
    callbacks.append(_checkpoint_callback())
    trainer_config["callbacks"] = callbacks
    trainer_config.setdefault("logger", logger_config)
    return trainer_config


def setup_runtime_logging(cfg: Config) -> None:
    if not is_rank_zero():
        logging.basicConfig(level=logging.WARNING)
        return
    setup_logging(cfg.save_dir)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def log_startup(cfg: Config, trainer_config: Dict[str, Any], loops_num: int) -> None:
    trainer_log_config = {
        key: value for key, value in trainer_config.items()
        if key not in {"callbacks", "logger"}
    }
    callback_names = ", ".join(
        type(callback).__name__ for callback in trainer_config["callbacks"]
    )
    logging.info("Config: %s", cfg.to_flat_dict())
    logging.info("Lightning trainer config: %s", trainer_log_config)
    logging.info("Lightning callbacks: %s", callback_names)
    logging.info("Lightning logger: %s", _logger_names(trainer_config["logger"]))
    logging.info("Cache loops per original epoch: %d", loops_num)


def _normalise_strategy(strategy: Any):
    if strategy == "ddp_find_unused_parameters_true" and DDPStrategy is not None:
        return DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=3600))
    return strategy


def _get_litlogger_class():
    for module_name in ("lightning.pytorch.loggers", "pytorch_lightning.loggers"):
        try:
            module = importlib.import_module(module_name)
            litlogger_cls = getattr(module, "LitLogger", None)
            if litlogger_cls is not None:
                return litlogger_cls
        except Exception:
            continue
    return None


def _metadata_to_strings(metadata: Any) -> Dict[str, str]:
    if not isinstance(metadata, dict):
        return {}
    return {str(key): str(value) for key, value in metadata.items()}


def _iter_loggers(logger_config: Any):
    if logger_config in (None, False):
        return []
    if isinstance(logger_config, (list, tuple)):
        return logger_config
    return [logger_config]


def _logger_names(logger_config: Any) -> str:
    if isinstance(logger_config, (list, tuple)):
        return ", ".join(type(logger).__name__ for logger in logger_config)
    return type(logger_config).__name__


def _disable_litlogger_graph_logging(logger: Any):
    if type(logger).__name__ != "LitLogger":
        return logger

    def _noop_log_graph(model: Any, input_array: Any = None) -> None:
        return None

    logger.log_graph = _noop_log_graph
    return logger


def _checkpoint_callback():
    return pl.callbacks.ModelCheckpoint(
        monitor="val/R_sum",
        mode="max",
        save_top_k=3,
        save_last=True,
        filename="epoch{epoch:03d}-r1{val/R@1:.2f}-rsum{val/R_sum:.2f}",
        auto_insert_metric_name=False,
    )
