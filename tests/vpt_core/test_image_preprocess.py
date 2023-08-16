import cv2
import numpy as np
import pytest
import shapely.geometry

from vpt_core.image import Header
from vpt_core.image.filter_factory import create_filter, create_filter_by_sequence
from vpt_core.segmentation.polygon_utils import (
    get_polygons_from_mask,
    get_upscale_matrix,
)


def test_empty() -> None:
    f = create_filter_by_sequence([])
    tmp = np.ones((1, 1), dtype=np.uint16)
    assert f is not None
    assert f(tmp)[0, 0] == 1
    f = create_filter_by_sequence(None)
    assert f is not None
    assert f(tmp)[0, 0] == 1


def test_invalid() -> None:
    with pytest.raises(NameError) as _:
        create_filter_by_sequence([Header("invalid", {"x": 1})])


@pytest.mark.parametrize("normalization_type", ["default", "CLAHE"], ids=str)
def test_normalize(normalization_type: str) -> None:
    f = create_filter(Header("normalize", {"type": normalization_type}))
    assert f is not None
    tmp = np.zeros((16, 16), dtype=np.uint16)
    zero_result = f(tmp)
    assert (zero_result == 0).all()

    tmp[1, 1] = 100
    result = f(tmp)
    assert result.dtype == np.uint8
    assert result[1, 1] == 255 and result[0, 0] == 0


def test_3d_clahe() -> None:
    filter_3d = create_filter(Header("normalize", {"type": "CLAHE", "image_dimensions": "3D"}))
    filter_2d = create_filter(Header("normalize", {"type": "CLAHE", "image_dimensions": "2D"}))
    tmp = np.zeros((2, 16, 16), dtype=np.uint16)
    tmp[0, 1, 1] = 100
    tmp[1, 1, 1] = 100
    result_3d = filter_3d([im for im in tmp])
    result_2d = filter_2d([im for im in tmp])
    assert isinstance(result_2d, list) and isinstance(result_3d, list)
    assert (np.array(result_2d) == np.array(result_3d)).all()
    assert result_3d[0][1, 1] == 255 and result_3d[0][0, 0] == 0
    assert result_3d[1][1, 1] == 255 and result_3d[1][0, 0] == 0


def test_unsupported_normalization() -> None:
    try:
        create_filter(Header("normalize", {"type": "random"}))
        assert False
    except TypeError:
        return


def test_blur() -> None:
    with pytest.raises(TypeError) as _:
        create_filter(Header("blur", {"type": "invalid"}))
    origin = np.zeros((5, 5), dtype=np.uint16)
    origin[2, 2] = 255
    f = create_filter(Header("blur", {"type": "median", "size": 3}))
    assert f is not None
    result = f(origin)
    assert result[2, 2] == 0
    f = create_filter(Header("blur", {"type": "gaussian", "size": 3}))
    assert f is not None
    result = f(origin)
    assert result[1, 1] > 0
    f = create_filter(Header("blur", {"type": "average", "size": 3}))
    assert f is not None
    result = f(origin)
    assert result[2, 2] == 28


def test_downsample() -> None:
    for scale in range(1, 8):
        f = create_filter(Header("downsample", {"scale": scale}))
        assert f is not None
        origin = np.zeros((100, 100), dtype=np.uint16)
        polygon = shapely.geometry.Polygon(
            [
                (15, 20),
                (30, 60),
                (20, 80),
                (40, 100),
                (90, 90),
                (90, 10),
                (60, 30),
                (35, 10),
            ]
        )
        cv2.fillPoly(origin, [np.array(polygon.exterior.xy).round().astype(np.int32).T], 1)
        result = f(origin)
        seg_res = get_polygons_from_mask((result != 0).astype("uint8"), 1, 1)
        seg_res.transform_geoms(get_upscale_matrix(scale, scale))
        upscaled = seg_res.get_z_geoms(0).tolist()
        assert len(upscaled) == 1
        upscaled = upscaled[0]
        assert polygon.symmetric_difference(upscaled).area < 100 * (scale * 2.3)


def test_merlin_filter() -> None:
    origin = np.array([[2] * 5] * 5, dtype=np.uint16)
    origin[2, 2] = 2000
    f = create_filter(Header("merlin-ws", {}))
    result = f(origin)
    assert result[2, 2] == 2000
    assert (result[:2, :2] == 0).all()


def test_sequence() -> None:
    filters = [Header("blur", {}), Header("normalize", {}), Header("blur", {"type": "median"})]
    origin = np.array([[2] * 5] * 5, dtype=np.uint16)
    origin[2, 2] = 230
    sequence = create_filter_by_sequence(filters)
    result = origin
    for filter_config in filters:
        f = create_filter(filter_config)
        result = f(result)
    assert (result == sequence(origin)).all()
