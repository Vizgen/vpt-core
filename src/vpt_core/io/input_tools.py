import geopandas as gpd
from pyarrow import parquet, Table
from shapely import wkb

from vpt_core.io.vzgfs import vzg_open, retrying_attempts
from vpt_core.segmentation.seg_result import SegmentationResult


DEFAULT_GEO_COLUMN_NAME = "geometry"


def pyarrow_table_to_pandas(table: Table):
    df = gpd.GeoDataFrame(table.to_pandas(integer_object_nulls=True))
    if SegmentationResult.geometry_field in df.columns:
        df[SegmentationResult.geometry_field] = df[SegmentationResult.geometry_field].apply(wkb.loads)
        df.set_geometry(SegmentationResult.geometry_field, inplace=True)
    else:
        df[DEFAULT_GEO_COLUMN_NAME] = df[DEFAULT_GEO_COLUMN_NAME].apply(wkb.loads)
    if SegmentationResult.parent_id_field in df.columns:
        df.loc[:, SegmentationResult.parent_id_field] = df.loc[:, SegmentationResult.parent_id_field].astype("object")
    return df


def read_parquet(path: str) -> gpd.GeoDataFrame:
    for attempt in retrying_attempts():
        with attempt, vzg_open(path, "rb") as f:
            pq = parquet.ParquetFile(f)
            return pyarrow_table_to_pandas(pq.read())
