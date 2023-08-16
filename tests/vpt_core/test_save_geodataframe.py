import os.path
import tempfile

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
import pytest

from vpt_core.io.input_tools import read_parquet
from vpt_core.io.output_tools import (
    save_geodataframe,
    save_geodataframe_with_row_groups,
    save_segmentation_results,
)


def test_empty():
    with tempfile.TemporaryDirectory() as td:
        output_path = str(td + "/" + "test_save_geodataframe_empty.geojson")
        save_geodataframe(gpd.GeoDataFrame(), output_path)
        assert gpd.read_file(output_path).empty


def get_geodataframe():
    geometry = gpd.GeoSeries.from_xy(*np.random.random((2, 500)))
    return gpd.GeoDataFrame(
        {
            "EntityID": np.random.random(500),
            "b": np.random.choice(list("abc"), size=500),
            "geometry": geometry,
        },
        index=np.random.permutation(range(500)),
    )


def test_row_groups():
    gdf = get_geodataframe()
    group_sizes = 10 + np.diff([0, *sorted(np.random.choice(range(400), size=9, replace=False)), 400])

    with tempfile.TemporaryDirectory() as td:
        output_path = str(td + "/" + "test_row_groups.parquet")
        save_geodataframe_with_row_groups(gdf, output_path, group_sizes)

        assert gdf.equals(read_parquet(output_path))

        with open(output_path, "rb") as f:
            file = pq.ParquetFile(f)

            read_group_sizes = np.array([len(file.read_row_group(i)) for i in range(file.num_row_groups)])

            assert all(group_sizes == read_group_sizes)


def test_write_parquet():
    gdf = get_geodataframe()

    with tempfile.TemporaryDirectory() as td:
        output_path = str(td + "/" + "test.parquet")
        save_geodataframe(gdf, output_path)
        assert gdf.equals(gpd.read_parquet(output_path))


def test_row_group_partitioning():
    gdf = get_geodataframe()
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "row_groups_test.parquet")
        save_segmentation_results(gdf, save_path, max_row_group_size=100)
        assert gdf.sort_index().equals(read_parquet(save_path).sort_index())


def test_save_geodataframe_bad_extension():
    gdf = get_geodataframe()
    with pytest.raises(ValueError):
        save_geodataframe(gdf, "fake_path.fake")
