import os.path
import tempfile
from typing import Set, Tuple, Optional

import numpy as np
import pytest

from tests.vpt_core import TEST_DATA_ROOT
from vpt_core.io.input_tools import read_parquet
from vpt_core.segmentation.seg_result import SegmentationResult
from vpt_core.utils.base_case import BaseCase
from vpt_core.utils.segmentation_utils import from_shapes_3d, Square


class RelationshipsCase(BaseCase):
    def __init__(
        self,
        name: str,
        child: SegmentationResult,
        parent: SegmentationResult,
        relations: Set[Tuple[Optional[int], Optional[int]]],
    ):
        super(RelationshipsCase, self).__init__(name)
        self.child = child
        self.parent = parent
        self.relations = relations


def get_id(subtract: int) -> np.int64:
    max_number = 1 << 63
    return max_number - subtract


@pytest.mark.parametrize(
    "case",
    [
        RelationshipsCase(
            "inside",
            from_shapes_3d([[Square(30, 30, 30), Square(50, 50, 30), Square(100, 50, 30)]], cids=[10, 11, 12]),
            from_shapes_3d([[Square(30, 30, 60), Square(50, 50, 60), Square(110, 50, 60)]], cids=[20, 21, 22]),
            {(20, 10), (21, 11), (22, 12)},
        ),
        RelationshipsCase(
            "parent_id_type",
            from_shapes_3d(
                [[Square(30, 30, 30), Square(50, 50, 30), Square(100, 50, 30)]], cids=[get_id(1), get_id(2), get_id(3)]
            ),
            from_shapes_3d(
                [[Square(30, 30, 60), Square(50, 50, 60), Square(110, 50, 60)]], cids=[get_id(4), get_id(5), get_id(6)]
            ),
            {(get_id(4), get_id(1)), (get_id(5), get_id(2)), (get_id(6), get_id(3))},
        ),
        RelationshipsCase(
            "no_parent",
            from_shapes_3d(
                [[Square(30, 30, 30)], [Square(70, 70, 30)], [Square(100, 50, 30)]],
                cids=[get_id(1), get_id(2), get_id(3)],
            ),
            from_shapes_3d([[Square(30, 30, 60)], [Square(110, 50, 60)]], cids=[get_id(4), get_id(5)]),
            {(get_id(4), get_id(1)), (None, get_id(2)), (get_id(5), get_id(3))},
        ),
        RelationshipsCase(
            "no_child",
            from_shapes_3d([[Square(30, 30, 30)], [Square(100, 50, 30)]], cids=[10, 11]),
            from_shapes_3d([[Square(30, 30, 60)], [Square(50, 50, 60)], [Square(110, 50, 60)]], cids=[20, 21, 22]),
            {(20, 10), (21, None), (22, 11)},
        ),
        RelationshipsCase(
            "several_children",
            from_shapes_3d(
                [[Square(30, 30, 30)], [Square(65, 60, 5)], [Square(80, 70, 5)], [Square(100, 50, 30)]],
                cids=[10, 11, 12, 13],
            ),
            from_shapes_3d([[Square(30, 30, 60)], [Square(110, 50, 60)]], cids=[20, 21]),
            {(20, 10), (20, 11), (20, 12), (21, 13)},
        ),
        RelationshipsCase(
            "cell inside nuclei",
            from_shapes_3d([[Square(30, 30, 60), Square(50, 50, 60), Square(110, 50, 60)]], cids=[10, 11, 12]),
            from_shapes_3d([[Square(30, 30, 30), Square(50, 50, 30), Square(100, 50, 30)]], cids=[20, 21, 22]),
            {(20, 10), (21, 11), (22, 12)},
        ),
        RelationshipsCase(
            "several cells inside nuclei",
            from_shapes_3d([[Square(30, 30, 10)], [Square(50, 50, 60)]], cids=[10, 11]),
            from_shapes_3d([[Square(30, 30, 5)], [Square(50, 50, 15)], [Square(68, 68, 20)]], cids=[20, 21, 22]),
            {(20, 10), (21, None), (22, 11)},
        ),
        RelationshipsCase(
            "cell overlapped other nuclei",
            from_shapes_3d([[Square(10, 10, 30), Square(10, 10, 40)], [Square(42, 42, 3)]], cids=[10, 10, 11]),
            from_shapes_3d([[Square(12, 12, 35)]], cids=[20]),
            {(None, 10), (20, 11)},
        ),
    ],
    ids=str,
)
def test_relationships(case: RelationshipsCase) -> None:
    case.child.create_relationships(case.parent, coverage_threshold=0.5)
    with tempfile.TemporaryDirectory(dir=str(TEST_DATA_ROOT)) as temp_dir:
        output = os.path.join(temp_dir, "test.parquet")
        case.child.df.to_parquet(output)
        res = read_parquet(output)
    assert all(res.dtypes == case.child.df.dtypes)
    if all(None not in pair for pair in case.relations):
        assert all(res[SegmentationResult.parent_id_field] >= 0)
        assert all(res[SegmentationResult.parent_entity_field] == case.parent.entity_type)
    assert len(case.relations) == max(
        len(res[SegmentationResult.cell_id_field].unique()),
        len(case.parent.df[SegmentationResult.cell_id_field].unique()),
    )
    for relation in case.relations:
        related = res[res[SegmentationResult.parent_id_field] == relation[0]]
        if relation[0] is None:
            assert any(res[SegmentationResult.parent_id_field].isnull())
            continue
        if relation[1] is None:
            assert len(related) == 0
            continue
        assert any(related[SegmentationResult.cell_id_field] == relation[1])
