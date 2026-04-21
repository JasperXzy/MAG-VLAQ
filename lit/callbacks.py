import logging
import time
from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import torch

try:
    import pytorch_lightning as pl
except ImportError as exc:  # pragma: no cover - depends on environment
    raise ImportError("PyTorch Lightning is required for callbacks.") from exc

from lit.ddp_utils import _barrier, broadcast_triplets


class _TripletProgress:
    def __init__(self, enabled=True, prefix="triplets", log_every=25, rich_progress=None):
        self.enabled = enabled
        self.prefix = prefix
        self.log_every = max(1, int(log_every))
        self._progress = rich_progress
        self._owns_progress = rich_progress is None
        self._tasks = {}
        self._summaries = []
        self._fallback = {}

    def __enter__(self):
        if not self.enabled:
            return self
        if self._progress is not None:
            return self
        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            commons = _load_lightning_commons()
            console = commons.get_rich_console()
            if console is None or not console.is_terminal:
                return self

            self._progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=True,
                refresh_per_second=2,
            )
            self._progress.start()
        except Exception:
            self._progress = None
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._progress is not None:
            console = self._progress.console
            if self._owns_progress:
                self._progress.stop()
            if self._summaries and console.is_terminal:
                width = max(len(summary["label"]) for summary in self._summaries)
                for summary in self._summaries:
                    console.print(
                        self._format_summary(summary, width, console.width),
                        highlight=False,
                        markup=False,
                    )
            if self._owns_progress:
                self._progress = None
        return False

    def __call__(self, event, desc, value):
        if not self.enabled:
            return

        label = f"{self.prefix}/{desc or 'stage'}"
        if self._progress is not None:
            self._handle_rich(event, label, value)
        else:
            self._handle_log(event, label, value)

    def _handle_rich(self, event, label, value):
        if event == "start":
            total = int(value) if value is not None else None
            task_id = self._progress.add_task(label, total=total)
            self._tasks[label] = {
                "id": task_id,
                "total": total,
                "current": 0,
                "t0": time.time(),
            }
        elif event == "advance":
            state = self._tasks.get(label)
            if state is None:
                return
            advance = int(value)
            state["current"] += advance
            self._progress.update(state["id"], advance=advance)
        elif event == "close":
            state = self._tasks.get(label)
            if state is None:
                return
            total = state["total"]
            if total is not None:
                self._progress.update(state["id"], completed=total, total=total)
            self._progress.stop_task(state["id"])
            if not self._owns_progress:
                self._progress.update(state["id"], visible=False)
            self._summaries.append(
                {
                    "label": label,
                    "current": total if total is not None else state["current"],
                    "total": total,
                    "elapsed": time.time() - state["t0"],
                }
            )

    @staticmethod
    def _format_duration(seconds):
        seconds = max(0, int(round(seconds)))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}"

    def _format_summary(self, summary, width, console_width):
        total = summary["total"]
        current = summary["current"]
        count = f"{current}/{total}" if total is not None else str(current)
        elapsed = self._format_duration(summary["elapsed"])
        prefix = f"{summary['label']:<{width}} "
        suffix = f" 100% {count} {elapsed} 0:00:00"
        bar_width = min(40, max(10, console_width - len(prefix) - len(suffix) - 1))
        bar = "\u2501" * bar_width
        return f"{prefix}{bar}{suffix}"

    def _handle_log(self, event, label, value):
        if event == "start":
            total = int(value) if value is not None else None
            self._fallback[label] = {
                "total": total,
                "current": 0,
                "next_report": self.log_every,
                "t0": time.time(),
            }
            logging.info("%s started%s", label, f" ({total} steps)" if total else "")
        elif event == "advance":
            state = self._fallback.get(label)
            if not state:
                return
            state["current"] += int(value)
            total = state["total"]
            if not total:
                return
            percent = min(100, int(state["current"] * 100 / total))
            if percent >= state["next_report"] and state["current"] < total:
                elapsed = time.time() - state["t0"]
                logging.info(
                    "%s %d%% (%d/%d, %.1fs)",
                    label,
                    percent,
                    state["current"],
                    total,
                    elapsed,
                )
                while state["next_report"] <= percent:
                    state["next_report"] += self.log_every
        elif event == "close":
            state = self._fallback.pop(label, None)
            if state and state["total"]:
                elapsed = time.time() - state["t0"]
                logging.info(
                    "%s done (%d/%d, %.1fs)",
                    label,
                    state["current"],
                    state["total"],
                    elapsed,
                )


def _set_dataset_progress_callback(dataset, callback):
    module = sys.modules.get(dataset.__class__.__module__)
    setter = getattr(module, "set_progress_callback", None) if module is not None else None
    if setter is None:
        return None
    setter(callback)
    return setter


def _load_lightning_commons():
    module_name = "_sca_lightning_commons"
    if module_name in sys.modules:
        return sys.modules[module_name]

    commons_path = Path(__file__).resolve().parents[1] / "commons.py"
    spec = spec_from_file_location(module_name, commons_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


def _hide_validation_progress(trainer):
    progress_bar = getattr(trainer, "progress_bar_callback", None)
    progress = getattr(progress_bar, "progress", None)
    val_task = getattr(progress_bar, "val_progress_bar_id", None)
    if progress is None or val_task is None:
        return
    try:
        progress.update(val_task, visible=False)
        refresh = getattr(progress_bar, "refresh", None)
        if refresh is not None:
            refresh()
    except Exception:
        pass


def _has_more_epochs(trainer):
    max_epochs = getattr(trainer, "max_epochs", None)
    if max_epochs is None or max_epochs < 0:
        return True
    return int(trainer.current_epoch) + 1 < int(max_epochs)


@contextmanager
def _paused_lightning_progress(trainer, resume=True):
    progress_bar = getattr(trainer, "progress_bar_callback", None)
    stop = getattr(progress_bar, "_stop_progress", None)
    init = getattr(progress_bar, "_init_progress", None)
    progress = getattr(progress_bar, "progress", None)
    if (
        progress_bar is None
        or progress is None
        or stop is None
        or getattr(progress_bar, "is_disabled", False)
    ):
        yield
        return

    stop()
    try:
        yield
    finally:
        if init is not None and resume:
            try:
                init(trainer)
            except Exception:
                pass


class TripletCacheRefreshCallback(pl.Callback):
    def __init__(self, loops_num):
        super().__init__()
        self.loops_num = loops_num
        self._last_refresh_epoch = None

    def on_fit_start(self, trainer, pl_module):
        self._refresh(trainer, pl_module, reason="fit_start")

    def on_train_epoch_start(self, trainer, pl_module):
        self._refresh(trainer, pl_module, reason="epoch_start")

    def _refresh(self, trainer, pl_module, reason):
        epoch = int(trainer.current_epoch)
        if self._last_refresh_epoch == epoch:
            return
        self._last_refresh_epoch = epoch

        dm = trainer.datamodule
        if getattr(dm, "triplets_ds", None) is None:
            dm.setup("fit")
        args = pl_module.args
        args.device = str(pl_module.device)
        real_epoch = epoch // self.loops_num
        loop = epoch % self.loops_num
        t0 = time.time()

        with _paused_lightning_progress(trainer, resume=True):
            logging.info(
                "compute triplets: lightning_epoch=%d real_epoch=%d loop=%d/%d reason=%s",
                epoch,
                real_epoch,
                loop,
                self.loops_num,
                reason,
            )

            old_disable_dataset_tqdm = getattr(args, "disable_dataset_tqdm", False)
            args.disable_dataset_tqdm = True
            dm.triplets_ds.is_inference = True
            setter = None
            try:
                with _TripletProgress(enabled=trainer.is_global_zero) as progress:
                    setter = _set_dataset_progress_callback(
                        dm.triplets_ds,
                        progress if trainer.is_global_zero else None,
                    )
                    dm.triplets_ds.compute_triplets(args, pl_module.model, pl_module.modelq)
            finally:
                if setter is not None:
                    setter(None)
                dm.triplets_ds.is_inference = False
                args.disable_dataset_tqdm = old_disable_dataset_tqdm

            if trainer.world_size > 1:
                _barrier(trainer.strategy, "triplet_cache_computed")
                broadcast_triplets(
                    dm.triplets_ds,
                    pl_module.device,
                    trainer.strategy,
                    trainer.world_size,
                    trainer.global_rank,
                )

            logging.info("triplet cache ready in %.2fs", time.time() - t0)


class RetrievalEvalCallback(pl.Callback):
    def __init__(self, loops_num):
        super().__init__()
        self.loops_num = loops_num
        self.best_r1r5r10ep = [0.0, 0.0, 0.0, 0]

    def on_validation_epoch_end(self, trainer, pl_module):
        import test

        args = pl_module.args
        args.device = str(pl_module.device)

        recalls_tensor = torch.zeros(3, device=pl_module.device, dtype=torch.float32)
        with _paused_lightning_progress(trainer, resume=_has_more_epochs(trainer)):
            _hide_validation_progress(trainer)
            if trainer.is_global_zero:
                logging.info(
                    "retrieval eval started: database=%d queries=%d batch_size=%d",
                    trainer.datamodule.test_ds.database_num,
                    trainer.datamodule.test_ds.queries_num,
                    args.infer_batch_size,
                )
            old_disable_dataset_tqdm = getattr(args, "disable_dataset_tqdm", False)
            old_test_progress_callback = getattr(test, "_TEST_PROGRESS_CALLBACK", None)
            args.disable_dataset_tqdm = True
            try:
                with _TripletProgress(
                    enabled=trainer.is_global_zero,
                    prefix="eval",
                ) as progress:
                    if hasattr(test, "set_progress_callback"):
                        test.set_progress_callback(progress if trainer.is_global_zero else None)
                    recalls, _, _ = test.test(
                        args,
                        trainer.datamodule.test_ds,
                        pl_module.model,
                        test_method=args.test_method,
                        modelq=pl_module.modelq,
                        rank=trainer.global_rank,
                        world_size=trainer.world_size,
                    )
            finally:
                if hasattr(test, "set_progress_callback"):
                    test.set_progress_callback(old_test_progress_callback)
                args.disable_dataset_tqdm = old_disable_dataset_tqdm

        if trainer.is_global_zero:
            recalls_tensor = torch.tensor(
                [float(recalls[0]), float(recalls[1]), float(recalls[2])],
                device=pl_module.device,
                dtype=torch.float32,
            )

        if trainer.world_size > 1:
            recalls_tensor = trainer.strategy.broadcast(recalls_tensor, src=0)

        current = [float(value) for value in recalls_tensor.detach().cpu().tolist()]
        real_epoch = int(trainer.current_epoch) // self.loops_num
        if sum(current) > sum(self.best_r1r5r10ep[:3]):
            self.best_r1r5r10ep = [*current, real_epoch]

        metrics = {
            "val/R@1": current[0],
            "val/R@5": current[1],
            "val/R@10": current[2],
            "val/R_sum": sum(current),
        }
        pl_module.log_dict(
            metrics,
            prog_bar=True,
            rank_zero_only=False,
            sync_dist=trainer.world_size > 1,
        )

        now = (
            f"Now : R@1 = {current[0]:.1f}   R@5 = {current[1]:.1f}   "
            f"R@10 = {current[2]:.1f}   epoch = {real_epoch:d}"
        )
        best = (
            f"Best: R@1 = {self.best_r1r5r10ep[0]:.1f}   "
            f"R@5 = {self.best_r1r5r10ep[1]:.1f}   "
            f"R@10 = {self.best_r1r5r10ep[2]:.1f}   "
            f"epoch = {self.best_r1r5r10ep[3]:d}"
        )
        if trainer.is_global_zero:
            commons = _load_lightning_commons()
            logging.info(now)
            logging.info(best)
            commons.logging_info(args, now)
            commons.logging_info(args, best)
