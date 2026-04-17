import logging
import os
import sys
import traceback
from functools import cache
from os.path import join

try:
    from rich import get_console, reconfigure
    from rich.logging import RichHandler
except ImportError:
    get_console = None
    reconfigure = None
    RichHandler = None


@cache
def get_rich_console():
    if get_console is None or reconfigure is None:
        return None
    try:
        reconfigure(stderr=True)
        return get_console()
    except Exception:
        return None


def setup_logging(
    save_dir,
    console="info",
    info_filename="info.log",
    debug_filename="debug.log",
):
    os.makedirs(save_dir, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s   %(message)s", "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

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

    if info_filename is not None:
        info_handler = logging.FileHandler(join(save_dir, info_filename))
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(formatter)
        logger.addHandler(info_handler)

    if debug_filename is not None:
        debug_handler = logging.FileHandler(join(save_dir, debug_filename))
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(formatter)
        logger.addHandler(debug_handler)

    if console is not None:
        rich_console = get_rich_console()
        if rich_console is not None and RichHandler is not None:
            console_handler = RichHandler(
                console=rich_console,
                show_time=True,
                show_level=False,
                show_path=False,
                markup=False,
                rich_tracebacks=False,
                log_time_format="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(logging.Formatter("%(message)s"))
        else:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.DEBUG if console == "debug" else logging.INFO)
        logger.addHandler(console_handler)

    def exception_handler(type_, value, tb):
        logger.info("\n" + "".join(traceback.format_exception(type_, value, tb)))

    sys.excepthook = exception_handler
