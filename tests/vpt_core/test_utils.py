import os
import tempfile

from shapely import box, geometry

from tests.vpt_core import TEST_DATA_ROOT
from vpt_core.io.input_tools import read_parquet
from vpt_core.segmentation.geometry_utils import convert_to_multipoly
from vpt_core.utils.copy_utils import _copy_between_filesystems, _copy_regex_images


def test_copying():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = str(TEST_DATA_ROOT / "cells.parquet")
        output_path = os.path.join(temp_dir, "cells.parquet")
        _copy_between_filesystems(input_path, output_path)
        assert all(read_parquet(input_path) == read_parquet(output_path))


def test_non_existent_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            _copy_between_filesystems(
                os.path.join(temp_dir, "non_exist.txt"),
                os.path.join(temp_dir, "new.txt"),
            )
            assert False
        except FileNotFoundError:
            return


def test_images_copying():
    with tempfile.TemporaryDirectory() as temp_dir:
        images_path = str(TEST_DATA_ROOT / r"input_images/mosaic_(?P<stain>[\w|-]+)_z(?P<z>[0])")
        _copy_regex_images(images_path, temp_dir, os.path.sep)
        assert len(os.listdir(temp_dir)) == 3


def test_non_existent_image_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        images_path = os.path.join(temp_dir, "{stain}_{z}")
        output_dir = os.path.join(temp_dir, "output")
        try:
            _copy_regex_images(images_path, output_dir, os.path.sep)
            assert False
        except FileNotFoundError:
            return


def test_convert_to_multipoly():
    poly = box(0, 0, 10, 10)
    multipoly = geometry.MultiPolygon([poly])
    assert convert_to_multipoly(poly) == multipoly
    assert convert_to_multipoly(multipoly) == multipoly

    multipoly2 = geometry.MultiPolygon([box(0, 0, 10, 10), box(11, 11, 16, 16)])
    collection = geometry.GeometryCollection([multipoly2, box(0, 0, 5, 5), geometry.Point(2, 3)])
    assert convert_to_multipoly(collection).symmetric_difference(multipoly2).area == 0
