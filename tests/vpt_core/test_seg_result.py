from dataclasses import asdict
from typing import Tuple, Set, List

import cv2
import numpy as np
import pytest
from shapely import MultiPolygon

from vpt_core.segmentation.polygon_utils import PolygonCreationParameters, generate_polygons_from_mask
from vpt_core.segmentation.seg_result import SegmentationResult
from vpt_core.segmentation.segmentation_item import difference
from vpt_core.utils.base_case import BaseCase
from vpt_core.utils.segmentation_utils import Square, from_shapes_3d, Rect, assert_seg_equals


def get_mask() -> Tuple[np.ndarray, MultiPolygon]:
    image = np.zeros((40, 40))
    poly = Square(20, 20, 10)
    cv2.fillPoly(image, [np.array(poly.geoms[0].exterior.xy).round().astype(np.int32).T], 1)
    return image, poly


def test_creation_from_mask() -> None:
    mask, poly = get_mask()
    params = PolygonCreationParameters(simplification_tol=1, smoothing_radius=1, minimum_final_area=10)
    seg_res = generate_polygons_from_mask(mask, asdict(params))
    assert len(seg_res.df) == 1
    assert seg_res.df.at[0, seg_res.geometry_field].symmetric_difference(poly).area < poly.area * 0.15


def test_cell_size_filter() -> None:
    seg_res = from_shapes_3d(
        [
            [Square(0, 0, 10 + x) for x in range(7)],
            [Square(0, 100, 20 + x) for x in range(7)],
            [Square(100, 0, 30 + x) for x in range(7)],
            [Square(100, 100, 40 + x) for x in range(7)],
        ]
    )
    seg_res.cell_size_filter(15 * 15)
    assert len(seg_res.df) == 7 * 4
    seg_res.cell_size_filter(400)
    assert len(seg_res.df) == 7 * 3
    seg_res.cell_size_filter(900)
    assert len(seg_res.df) == 7 * 2


def test_edge_polys_removal() -> None:
    seg_res = from_shapes_3d([[Square(30, 30, 30), Square(50, 50, 30), Square(100, 50, 30)]], cids=[0, 1, 1])
    window = [120, 120]
    seg_res.remove_edge_polys(window)
    assert len(seg_res.df) == 1


class ZReplicationCase(BaseCase):
    def __init__(self, name: str, planes: List[int], result: Set[Tuple[int, int]]):
        super(ZReplicationCase, self).__init__(name)
        self.planes = planes
        self.result = result


@pytest.mark.parametrize(
    "case",
    [
        ZReplicationCase("neighbor", [0, 3, 6], {(1, 0), (2, 3), (4, 3), (5, 6)}),
        ZReplicationCase("shift1", [0, 2, 4], {(1, 2), (3, 4)}),
    ],
    ids=str,
)
def test_z_replication_(case: ZReplicationCase) -> None:
    shape = Square(0, 0, 2)
    seg_res = from_shapes_3d([[shape] * 3])
    seg_res.update_column(seg_res.z_index_field, lambda i: case.planes[i])
    z_planes = max(case.planes) + 1
    assert set(seg_res._get_replication_pairs(set(range(z_planes)))) == case.result

    seg_res.replicate_across_z(list(range(z_planes)))
    assert len(seg_res.df) == z_planes
    assert set(seg_res.df[seg_res.z_index_field]) == set(range(z_planes))

    def get_z_geometries(z_index: int):
        return seg_res.df[seg_res.df[seg_res.z_index_field] == z_index][seg_res.geometry_field].to_list()

    for pair in case.result:
        assert get_z_geometries(pair[0]) == get_z_geometries(pair[1])


def test_update_splitted_item_difference() -> None:
    s1 = from_shapes_3d([[Square(0, 0, 10), Square(5, 5, 10)], [Square(1, 1, 10), Square(1, 1, 20)]])

    s2 = from_shapes_3d([[Square(5, 0, 11), Square(10, 5, 11)], [Square(1, 1, 10), Square(1, 1, 20)]])
    result = difference(s1.item(0), s2.item(0), 0.01)
    connected = SegmentationResult.get_connected_items(result)
    assert len(connected) == 2
    s1.update_item(connected[0])
    item_id = s1.add_item(connected[1])

    # set new id for geometry at z plane 1
    gt_seg = from_shapes_3d([[Rect(0, 0, 5, 10), Rect(5, 5, 5, 10)], [Square(1, 1, 10), Square(1, 1, 20)]])
    updated_row = (gt_seg.df[gt_seg.cell_id_field] == 0) * (gt_seg.df[gt_seg.z_index_field] == 1)
    gt_seg.df[updated_row] = gt_seg.df[updated_row].assign(**{gt_seg.cell_id_field: item_id})
    assert_seg_equals(s1, gt_seg, 1)


def test_3d_splitted_difference() -> None:
    y_len = 5
    s1 = from_shapes_3d(
        [
            [
                Rect(0, 0, 25, y_len),
                Rect(0, 0, 7, y_len).union(Rect(18, 0, 7, y_len)),
                Rect(20, 0, 5, y_len),
                Rect(0, 0, 7, y_len).union(Rect(18, 0, 7, y_len)),
                Rect(0, 0, 25, y_len),
            ]
        ]
    )
    s2 = from_shapes_3d([[Rect(10, 0, 5, y_len)]])

    result = difference(s1.item(0), s2.item(0), 0.01)
    connected = SegmentationResult.get_connected_items(result)
    assert len(connected) == 2
    s1.update_item(connected[0])
    item_id = s1.add_item(connected[1])

    gt_seg = from_shapes_3d(
        [
            [Rect(0, 0, 10, y_len), Rect(0, 0, 7, y_len)],
            [
                Rect(15, 0, 10, y_len),
                Rect(18, 0, 7, y_len),
                Rect(20, 0, 5, y_len),
                Rect(0, 0, 7, y_len).union(Rect(18, 0, 7, y_len)),
                Rect(0, 0, 25, y_len),
            ],
        ],
        cids=[item_id, item_id, 0, 0, 0, 0, 0],
    )

    assert_seg_equals(s1, gt_seg, 1)
