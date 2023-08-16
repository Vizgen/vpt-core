from typing import List, Optional

import numpy as np
import pandas as pd
from shapely import MultiPolygon, box

from vpt_core.segmentation.seg_result import SegmentationResult


def Square(x, y, a) -> MultiPolygon:
    return MultiPolygon([box(x, y, x + a, y + a)])


def Rect(x, y, a, b) -> MultiPolygon:
    return MultiPolygon([box(x, y, x + a, y + b)])


def from_shapes(
    geoms: List[MultiPolygon], cids: Optional[List[int]] = None, entity_type: str = "cell"
) -> SegmentationResult:
    ids = range(len(geoms))
    res = SegmentationResult(
        list_data={
            SegmentationResult.detection_id_field: list(ids),
            SegmentationResult.z_index_field: [0 for _ in ids],
            SegmentationResult.cell_id_field: cids if cids else list(ids),
            SegmentationResult.geometry_field: geoms,
        },
        entity=entity_type,
    )
    res.set_column(res.entity_name_field, res.entity_type)
    return res


def from_shapes_3d(
    geoms: List[List[MultiPolygon]], cids: Optional[List[int]] = None, entity_type: str = "cell"
) -> SegmentationResult:
    zs = [z for cell in geoms for z in range(len(cell))]
    ci = cids if cids else [cell_id for cell_id, cell in enumerate(geoms) for _ in cell]
    res = SegmentationResult(
        list_data={
            SegmentationResult.detection_id_field: list(range(sum(len(x) for x in geoms))),
            SegmentationResult.z_index_field: zs,
            SegmentationResult.cell_id_field: ci,
            SegmentationResult.geometry_field: [g for cell in geoms for g in cell],
        },
        entity=entity_type,
    )
    res.set_column(res.entity_name_field, res.entity_type)
    return res


def assert_df_equals(gdf1, gdf2, area_epsilon: float = 100):
    gdf1_sorted = gdf1.sort_values([SegmentationResult.cell_id_field, SegmentationResult.z_index_field])
    gdf2_sorted = gdf2.sort_values([SegmentationResult.cell_id_field, SegmentationResult.z_index_field])

    iloc1 = gdf1_sorted.drop(
        [SegmentationResult.geometry_field, SegmentationResult.detection_id_field],
        axis=1,
    ).reset_index(drop=True)
    iloc2 = gdf2_sorted.drop(
        [SegmentationResult.geometry_field, SegmentationResult.detection_id_field],
        axis=1,
    ).reset_index(drop=True)

    assert np.logical_or(iloc1 == iloc2, np.logical_and(pd.isnull(iloc1), pd.isnull(iloc2))).all().all()
    v1 = gdf1_sorted[SegmentationResult.geometry_field].values
    v2 = gdf2_sorted[SegmentationResult.geometry_field].values
    assert (v1.symmetric_difference(v2).area < area_epsilon).all()


def assert_seg_equals(s1: SegmentationResult, s2: SegmentationResult, area_epsilon: float = 100):
    assert_df_equals(s1.df, s2.df, area_epsilon)
