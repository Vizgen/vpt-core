from time import time

from vpt_core.io import output_tools


def test_format_experiment_timestamp():
    current_time = time()
    x = output_tools.format_experiment_timestamp(current_time)
    assert type(x) is str
    assert int(x) < 1e8
