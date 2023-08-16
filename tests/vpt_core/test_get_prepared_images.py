from typing import List

import numpy as np
import pytest

from vpt_core.io.image import ImageSet, get_prepared_images
from vpt_core.segmentation.segmentation_task import SegTask
from vpt_core.segmentation.task_input_data import InputData


def construct_simple_test_seg_task(inputs: List[InputData]) -> SegTask:
    return SegTask(0, "test", ["test"], [0], inputs, {}, {}, {})


def construct_3d_test_seg_task(inputs: List[InputData]) -> SegTask:
    return SegTask(0, "test", ["test"], [0, 1], inputs, {}, {}, {})


@pytest.fixture
def image_set() -> ImageSet:
    ret = ImageSet()
    ret["CH"] = {0: np.ones((256, 256), dtype=np.uint16)}
    yield ret


@pytest.fixture
def image_set_3d() -> ImageSet:
    ret = ImageSet()
    ret["CH"] = {1: np.ones((256, 256), dtype=np.uint16), 0: np.ones((256, 256), dtype=np.uint16)}
    yield ret


def test_get_prepared_images_no_filters(image_set: ImageSet) -> None:
    result, scale = get_prepared_images(construct_simple_test_seg_task([InputData("CH", [])]), image_set)
    assert scale == (1.0, 1.0)
    assert result.as_stack().shape == (1, 256, 256, 1)


def test_get_prepared_images_channel(image_set: ImageSet) -> None:
    # no images for requested channel
    with pytest.raises(KeyError):
        get_prepared_images(construct_simple_test_seg_task([InputData("CH", [])]), ImageSet())

    # invalid channel name
    with pytest.raises(KeyError):
        get_prepared_images(
            SegTask(0, "test", ["test"], [0], [InputData("invalid", [])], {}, {}, {}),
            image_set,
        )


def test_get_prepared_images_invalid_filters(image_set) -> None:
    # invalid json
    with pytest.raises(TypeError):
        get_prepared_images(
            construct_simple_test_seg_task([InputData("CH", [{"a": None}])]),
            image_set,
        )

    # invalid filter
    with pytest.raises(NameError):
        get_prepared_images(
            construct_simple_test_seg_task([InputData("CH", [{"name": "invalid"}])]),
            image_set,
        )


def test_get_prepared_images_filters(image_set) -> None:
    # downsample
    downsamle_json = {"name": "downsample", "parameters": {"scale": 2}}
    result, scale = get_prepared_images(construct_simple_test_seg_task([InputData("CH", [downsamle_json])]), image_set)
    assert scale == (2.0, 2.0)
    assert result.as_stack().shape == (1, 128, 128, 1)
    # multiple filters
    norm_json = {"name": "normalize"}
    blur_json = {"name": "blur", "parameters": {"type": "average", "size": 5}}
    merlin_ws_json = {"name": "merlin-ws"}
    result, scale = get_prepared_images(
        construct_simple_test_seg_task([InputData("CH", [norm_json, blur_json, merlin_ws_json])]),
        image_set,
    )
    assert scale == (1.0, 1.0)
    assert result.as_stack().shape == (1, 256, 256, 1)


def test_get_prepared_images_3d(image_set_3d) -> None:
    norm_json = {"name": "normalize", "parameters": {"type": "CLAHE", "image_dimensions": "3D"}}
    blur_json = {"name": "blur"}
    merlin_ws_json = {"name": "merlin-ws"}
    result, scale = get_prepared_images(
        construct_3d_test_seg_task([InputData("CH", [norm_json, blur_json, merlin_ws_json])]),
        image_set_3d,
    )
    assert scale == (1.0, 1.0)
    assert result.as_stack().shape == (2, 256, 256, 1)
