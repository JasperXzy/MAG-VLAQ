import argparse
import copy
import importlib
import importlib.util
import logging
import math
import os
from datetime import timedelta
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple
import warnings

import torch

try:
    import pytorch_lightning as pl
except ImportError as exc:  # pragma: no cover - depends on environment
    raise ImportError(
        "PyTorch Lightning is required for the Lightning training entrypoint. "
        "Install pytorch-lightning before running lightning/train.py."
    ) from exc

try:
    from pytorch_lightning.strategies import DDPStrategy
except Exception:  # pragma: no cover - older Lightning versions
    DDPStrategy = None

try:
    from pytorch_lightning.cli import LightningCLI as _BaseLightningCLI
except Exception:  # pragma: no cover - older Lightning versions
    try:
        from pytorch_lightning.utilities.cli import LightningCLI as _BaseLightningCLI
    except Exception:
        _BaseLightningCLI = object


LIGHTNING_COMMANDS = {"fit", "train", "validate", "test", "predict"}
RESERVED_CONFIG_KEYS = {"trainer", "model", "data", "ckpt_path"}
STORE_TRUE_FLAGS = {"horizontal_flip", "efficient_ram_testing"}


def _suppress_noisy_runtime_warnings():
    warning_patterns = [
        r".*xFormers is not available.*",
        r".*is_fx_tracing.*",
        r".*isinstance\(treespec, LeafSpec\).*",
        r".*Found .* module\(s\) in eval mode at the start of training.*",
        r".*val_dataloader.*does not have many workers.*",
        r".*tensorboardX.*has been removed as a dependency.*",
    ]
    for pattern in warning_patterns:
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


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("PyYAML is required when using --config YAML files.") from exc

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _parse_scalar(value: str) -> Any:
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _set_nested(config: Dict[str, Any], dotted_key: str, value: Any):
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _consume_value(argv: List[str], index: int) -> Tuple[Any, int]:
    token = argv[index]
    if "=" in token:
        return _parse_scalar(token.split("=", 1)[1]), index + 1
    if index + 1 >= len(argv):
        return True, index + 1
    return _parse_scalar(argv[index + 1]), index + 2


def _split_cli(argv: Iterable[str]) -> Tuple[str, List[str], Dict[str, Any], str, List[str]]:
    argv = list(argv)
    command = "fit"
    if argv and argv[0] in LIGHTNING_COMMANDS:
        command = argv.pop(0)

    config_paths: List[str] = []
    inline_config: Dict[str, Any] = {}
    ckpt_path = None
    legacy_args: List[str] = []

    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--config":
            config_paths.append(argv[i + 1])
            i += 2
        elif token.startswith("--config="):
            config_paths.append(token.split("=", 1)[1])
            i += 1
        elif token == "--ckpt_path":
            ckpt_path = argv[i + 1]
            i += 2
        elif token.startswith("--ckpt_path="):
            ckpt_path = token.split("=", 1)[1]
            i += 1
        elif token.startswith("--trainer."):
            key = token[len("--trainer.") :].split("=", 1)[0]
            value, i = _consume_value(argv, i)
            _set_nested(inline_config, f"trainer.{key}", value)
        elif token.startswith("--model."):
            key = token[len("--model.") :].split("=", 1)[0]
            value, i = _consume_value(argv, i)
            _set_nested(inline_config, f"model.{key}", value)
        elif token.startswith("--data."):
            key = token[len("--data.") :].split("=", 1)[0]
            value, i = _consume_value(argv, i)
            _set_nested(inline_config, f"data.{key}", value)
        else:
            legacy_args.append(token)
            i += 1

    return command, config_paths, inline_config, ckpt_path, legacy_args


def _flatten_legacy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    legacy: Dict[str, Any] = {}
    for key, value in config.items():
        if key not in RESERVED_CONFIG_KEYS:
            legacy[key] = value
    for section in ("model", "data"):
        section_values = config.get(section, {})
        if section_values:
            legacy.update(section_values)
    return legacy


def _value_to_argv(key: str, value: Any) -> List[str]:
    if value is None:
        return []
    if key == "mining" and value == "partial_sep":
        # tools/options.py has partial_sep as the default, but forgot to add it
        # to argparse choices. Omitting it preserves the intended legacy value.
        return []
    flag = f"--{key}"
    if key in STORE_TRUE_FLAGS:
        return [flag] if bool(value) else []
    if isinstance(value, bool):
        return [flag, "True" if value else "False"]
    if isinstance(value, (list, tuple)):
        return [flag, *[str(v) for v in value]]
    if isinstance(value, str):
        value = os.path.expanduser(value)
    return [flag, str(value)]


def _config_to_legacy_argv(config: Dict[str, Any]) -> List[str]:
    argv: List[str] = []
    for key, value in _flatten_legacy_config(config).items():
        argv.extend(_value_to_argv(key, value))
    return argv


def _parse_legacy_args(legacy_argv: List[str]) -> argparse.Namespace:
    old_argv = sys.argv[:]
    sys.argv = [old_argv[0], *legacy_argv]
    from tools import options as parser

    args = parser.parse_arguments()
    # Keep sys.argv in the legacy-compatible form for code that still reads argv.
    return args


def _import_symbol(target: str):
    module_name, symbol = target.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol)


def _load_lightning_commons():
    module_name = "_sca_lightning_commons"
    if module_name in sys.modules:
        return sys.modules[module_name]

    commons_path = Path(__file__).resolve().parents[1] / "commons.py"
    spec = importlib.util.spec_from_file_location(module_name, commons_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _is_rank_zero() -> bool:
    return _rank() == 0


def _normalise_strategy(strategy: Any):
    if strategy == "ddp_find_unused_parameters_true" and DDPStrategy is not None:
        return DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=3600))
    return strategy


def _default_precision(args: argparse.Namespace) -> str:
    amp_dtype = getattr(args, "amp_dtype", "none")
    if amp_dtype == "bf16":
        return "bf16-mixed"
    if amp_dtype == "fp16":
        return "16-mixed"
    return "32-true"


class SCALightningCLI(_BaseLightningCLI):
    """Small Lightning entrypoint that preserves the repo's legacy argparse contract.

    Lightning-style YAML is converted into the legacy argparse namespace at the
    entrypoint, then passed explicitly into the LightningModule/DataModule.
    """

    def __init__(self, model_class: str, datamodule_class: str):
        _suppress_noisy_runtime_warnings()
        raw_argv = sys.argv[1:]
        command, config_paths, inline_config, ckpt_path, legacy_cli_args = _split_cli(raw_argv)

        config: Dict[str, Any] = {}
        for path in config_paths:
            _deep_update(config, _load_yaml(path))
        _deep_update(config, inline_config)
        ckpt_path = ckpt_path if ckpt_path is not None else config.get("ckpt_path")

        legacy_argv = [*_config_to_legacy_argv(config), *legacy_cli_args]
        args = _parse_legacy_args(legacy_argv)
        if getattr(args, "save_dir", "default") == "default":
            args.save_dir = os.path.join("logs", args.exp_name)
        loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        trainer_config = copy.deepcopy(config.get("trainer", {}))
        trainer_config.setdefault("accelerator", "gpu" if torch.cuda.is_available() else "auto")
        trainer_config.setdefault("devices", "auto")
        trainer_config.setdefault("precision", _default_precision(args))
        trainer_config.setdefault("reload_dataloaders_every_n_epochs", 1)
        trainer_config.setdefault("check_val_every_n_epoch", loops_num)
        trainer_config.setdefault("max_epochs", args.epochs_num * loops_num)
        trainer_config.setdefault("default_root_dir", args.save_dir)
        trainer_config.setdefault("num_sanity_val_steps", 0)
        trainer_config.setdefault("log_every_n_steps", 20)
        trainer_config.setdefault("enable_model_summary", False)
        trainer_config.setdefault("sync_batchnorm", torch.cuda.is_available())
        trainer_config["strategy"] = _normalise_strategy(
            trainer_config.get("strategy", "auto")
        )

        pl.seed_everything(args.seed + _rank(), workers=True)
        self._setup_text_logging(args)

        model_cls = _import_symbol(model_class)
        datamodule_cls = _import_symbol(datamodule_class)
        from lit.callbacks import RetrievalEvalCallback, TripletCacheRefreshCallback

        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            monitor="val/R_sum",
            mode="max",
            save_top_k=3,
            save_last=True,
            filename="epoch{epoch:03d}-r1{val/R@1:.2f}-rsum{val/R_sum:.2f}",
            auto_insert_metric_name=False,
        )
        callbacks = trainer_config.pop("callbacks", None) or []
        has_progress_bar = any(
            type(callback).__name__.endswith("ProgressBar") for callback in callbacks
        )
        callbacks.append(TripletCacheRefreshCallback(loops_num=loops_num))
        if trainer_config.get("enable_progress_bar", True) and not has_progress_bar:
            callbacks.append(pl.callbacks.RichProgressBar(leave=True, console_kwargs={"stderr": True}))
        callbacks.extend(
            [
                RetrievalEvalCallback(loops_num=loops_num),
                checkpoint_cb,
            ]
        )
        trainer_config["callbacks"] = callbacks
        trainer_config.setdefault(
            "logger",
            pl.loggers.CSVLogger(save_dir=args.save_dir, name="lightning_logs"),
        )

        model = model_cls(args)
        datamodule = datamodule_cls(args)
        trainer = pl.Trainer(**trainer_config)

        trainer_log_config = {
            key: value for key, value in trainer_config.items()
            if key not in {"callbacks", "logger"}
        }
        callback_names = ", ".join(type(callback).__name__ for callback in callbacks)
        logging.info("Arguments: %s", args)
        logging.info("Lightning trainer config: %s", trainer_log_config)
        logging.info("Lightning callbacks: %s", callback_names)
        logging.info("Lightning logger: %s", type(trainer_config["logger"]).__name__)
        logging.info("Cache loops per original epoch: %d", loops_num)

        if command not in {"fit", "train"}:
            raise ValueError(f"Only the 'fit'/'train' command is implemented, got: {command}")
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

        if _is_rank_zero():
            commons = _load_lightning_commons()
            commons.logging_end(args)

    @staticmethod
    def _setup_text_logging(args: argparse.Namespace):
        if not _is_rank_zero():
            logging.basicConfig(level=logging.WARNING)
            return

        commons = _load_lightning_commons()

        commons.setup_logging(args.save_dir)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        commons.logging_init(args)
