import random
from unittest.mock import patch

from fsspec import FSTimeoutError
from tenacity import RetryError

from tests.vpt_core import TEST_DATA_ROOT
from vpt_core.io.vzgfs import vzg_open, retrying_attempts, io_with_retries


def random_throw_timeout(*args, **kwargs):
    if "calls" not in random_throw_timeout.__dict__:
        random_throw_timeout.calls = 0
    random_throw_timeout.calls += 1
    probability = [100, 50, 25, 10, 0]
    if random_throw_timeout.calls + 1 > len(probability):
        return
    if random.uniform(1, 100) < probability[random_throw_timeout.calls - 1]:
        raise FSTimeoutError("timeout!")


@patch("os.read", side_effect=random_throw_timeout)
def random_read_file(f, mock_read):
    mock_read(f, 4)


@patch("os.read", side_effect=FSTimeoutError("timeout!"))
def failed_read_file(f, mock_read):
    mock_read(f, 4)


def test_timeout_retry() -> None:
    for attempt in retrying_attempts():
        with attempt, vzg_open(str(TEST_DATA_ROOT / "cells.parquet"), "rb") as f:
            random_read_file(f)


def test_io_operation_retry() -> None:
    io_with_retries(str(TEST_DATA_ROOT / "cells.parquet"), "rb", random_read_file)


def test_retry_failed() -> None:
    try:
        for attempt in retrying_attempts():
            with attempt, vzg_open(str(TEST_DATA_ROOT / "cells.parquet"), "rb") as f:
                failed_read_file(f)
        assert False
    except RetryError:
        pass
