import logging
import time
from contextlib import contextmanager

try:
    import pytorch_lightning as pl
except ImportError as exc:  # pragma: no cover - depends on environment
    raise ImportError("PyTorch Lightning is required for callbacks.") from exc

from sca.lightning.logging_utils import get_rich_console
from sca.lightning.triplet_cache import TripletCacheBuilder


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

            console = get_rich_console()
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
        cfg = pl_module.cfg
        cfg.device = str(pl_module.device)
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

            dm.triplets_ds.is_inference = True
            try:
                with _TripletProgress(enabled=trainer.is_global_zero) as progress:
                    builder = TripletCacheBuilder(
                        cfg,
                        trainer,
                        pl_module,
                        dm.triplets_ds,
                    )
                    dm.triplets_ds.triplets_global_indexes = builder.refresh(
                        progress if trainer.is_global_zero else None
                    )
            finally:
                dm.triplets_ds.is_inference = False

            logging.info("triplet cache ready in %.2fs", time.time() - t0)
