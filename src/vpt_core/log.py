import logging
import pathlib
import sys
from typing import List

from tqdm import tqdm

_t_id = "/"


class VPTFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.tid = _t_id
        record.exc_text, record.exc_info, record.stack_info = None, None, None
        return super().format(record)


vpt_logger = logging.getLogger("vpt")
logger: logging.Logger = vpt_logger
h_stdout = logging.StreamHandler(sys.stdout)
formatter = VPTFormatter("%(asctime)s - %(tid)s - %(levelname)s - %(message)s")
h_stdout.setFormatter(formatter)
file_handlers: List[logging.FileHandler] = []


def set_logger(user_logger: logging.Logger):
    global logger, vpt_logger
    vpt_logger = logger
    logger = user_logger
    set_verbose(True)


def release_logger():
    global logger, vpt_logger, file_handlers
    for fh in file_handlers:
        logger.removeHandler(fh)
        fh.close()
    file_handlers = []
    logger = vpt_logger


def set_process_name(proc_name: str):
    global _t_id
    _t_id = proc_name


def initialize_logger(fname: str, lvl: int = 1, verbose: bool = False) -> None:
    logger.setLevel(lvl * 10)
    if fname:
        pathlib.Path(fname).parent.mkdir(parents=True, exist_ok=True)
        set_log_file(fname)
    if lvl > 1:
        # todo: find a better place for this dask-related calls:
        dask_logger = logging.getLogger("distributed.worker")
        dask_logger.setLevel(logging.ERROR)
    set_verbose(verbose)


def set_log_file(fname: str) -> None:
    fh = logging.FileHandler(filename=fname)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    file_handlers.append(fh)


def set_verbose(v: bool = True) -> None:
    std_in = is_verbose()
    if std_in == v:
        return
    if v:
        logger.addHandler(h_stdout)
    else:
        logger.removeHandler(h_stdout)


def is_verbose() -> bool:
    return h_stdout in logger.handlers


def show_progress(iterable, *args, **kwargs):
    return iterable if not is_verbose() else tqdm(iterable, *args, **kwargs)


def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs)


def critical(msg, *args, **kwargs):
    logger.critical(msg, *args, **kwargs)


def log_system_info():
    import psutil
    from psutil._common import bytes2human

    stat = psutil.virtual_memory()

    text = [f"{name}: {bytes2human(getattr(stat, name))}" for name in ["total", "available", "used", "free"]]
    debug(f"memory stat:{', '.join(text)}")
