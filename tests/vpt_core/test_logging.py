import io
import logging
import os.path
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator

import pytest

import vpt_core.log as log
from tests.vpt_core import TEST_DATA_ROOT


@pytest.fixture
def output_folder() -> Generator:
    with TemporaryDirectory(dir=str(TEST_DATA_ROOT)) as td:
        yield Path(td)


def test_user_logger(output_folder: str) -> None:
    user_logger = logging.getLogger("own")
    user_log_file = os.path.join(output_folder, "log.txt")
    message = "test_user_logger"

    fh = logging.FileHandler(filename=user_log_file)
    user_logger.addHandler(fh)
    user_logger.setLevel(10)

    log.set_logger(user_logger)
    log.set_verbose(False)
    log_methods = [log.info, log.warning, log.error, log.debug, log.critical]
    for log_method in log_methods:
        log_method(message)

    with open(user_log_file, "r") as f:
        assert f.read() == "\n".join([message] * len(log_methods)) + "\n"

    log.release_logger()
    fh.close()

    assert log.logger != user_logger
    assert log.logger == log.vpt_logger


def test_file_logging(output_folder: str):
    log_file = os.path.join(output_folder, "log.txt")
    process_name = "test_process_name"
    log.initialize_logger(log_file, lvl=1)
    log.set_verbose(False)
    log.set_process_name(process_name)

    log.log_system_info()
    with open(log_file, "r") as f:
        data = f.read()
        assert data.find("memory stat:") > 0
        assert data.find("DEBUG") > 0
        assert data.find(process_name) > 0
    log.release_logger()


def test_logging_progress(output_folder: str):
    data = range(5)
    log_file = os.path.join(output_folder, "log.txt")
    log.initialize_logger(log_file, verbose=True)
    with io.StringIO() as f:
        for i, x in enumerate(log.show_progress(data, file=f)):
            assert x == data[i]
        assert f.getvalue().find("100%") > 0
    log.release_logger()
