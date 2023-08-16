import pytest

from vpt_core.segmentation.segmentation_item import intersection, difference
from vpt_core.utils.segmentation_utils import Square, from_shapes_3d, assert_seg_equals, Rect


def test_get_item() -> None:
    storage = from_shapes_3d([[Square(0, 0, 10), Square(2, 2, 4)], [Square(15, 15, 10), Square(20, 20, 2)]])
    item = storage.item(0)
    assert item.get_entity_type() == "cell"
    assert item.get_entity_id() == 0
    assert item.get_volume() == 116
    with pytest.raises(KeyError):
        storage.item(4)
    item = storage.item(1)
    assert item.get_entity_id() == 1
    assert item.get_volume() == 104


def test_update_item_intersection() -> None:
    s1 = from_shapes_3d([[Square(0, 0, 10), Square(5, 5, 10)], [Square(100, 100, 10), Square(105, 105, 5)]])

    s2 = from_shapes_3d([[Square(5, 0, 11), Square(10, 5, 11)], [Square(110, 110, 10), Square(115, 115, 5)]])
    result = intersection(s1.item(0), s2.item(0))
    s1.update_item(result)
    assert_seg_equals(
        s1, from_shapes_3d([[Rect(5, 0, 5, 10), Rect(10, 5, 5, 10)], [Square(100, 100, 10), Square(105, 105, 5)]]), 1
    )


def test_item_intersection_different_z_planes() -> None:
    s1 = from_shapes_3d([[Square(0, 0, 10), Square(5, 5, 10)]])

    s2 = from_shapes_3d([[Square(5, 0, 11)]])
    result = intersection(s1.item(0), s2.item(0))
    s1.update_item(result)
    assert_seg_equals(s1, from_shapes_3d([[Rect(5, 0, 5, 10)]]), 1)


def test_update_item_difference() -> None:
    s1 = from_shapes_3d([[Square(0, 0, 10), Square(5, 5, 10)], [Square(1, 1, 10), Square(1, 1, 20)]])

    s2 = from_shapes_3d([[Square(9, 0, 11), Square(10, 5, 11)], [Square(1, 1, 10), Square(1, 1, 20)]])
    result = difference(s1.item(0), s2.item(0), 0.01)
    s1.update_item(result)
    assert_seg_equals(
        s1, from_shapes_3d([[Rect(0, 0, 9, 10), Rect(5, 5, 5, 10)], [Square(1, 1, 10), Square(1, 1, 20)]]), 1
    )


def test_update_item_empty() -> None:
    s1 = from_shapes_3d([[Square(0, 0, 10)], [Square(0, 0, 10), Square(1, 1, 10)]])

    s2 = from_shapes_3d([[Square(5, 0, 11)], [Square(5, 0, 11), Square(1, 1, 10)]])
    result = difference(s1.item(1), s2.item(1), 0.01)
    s1.update_item(result)
    assert_seg_equals(s1, from_shapes_3d([[Square(0, 0, 10)], [Rect(0, 0, 5, 10)]]), 1)


def test_create_item() -> None:
    s1 = from_shapes_3d([[Square(0, 0, 10), Square(5, 5, 10)]])
    s2 = from_shapes_3d([[Square(1, 1, 10), Square(1, 1, 20)]])
    new_id = s1.add_item(s2.item(0))
    assert new_id == 1
    assert_seg_equals(
        s1,
        from_shapes_3d([[Square(0, 0, 10), Square(5, 5, 10)], [Square(1, 1, 10), Square(1, 1, 20)]], [0, 0, 1, 1]),
        1,
    )


def test_set_parent() -> None:
    s1 = from_shapes_3d([[Square(0, 0, 10), Square(5, 5, 10)], [Square(1, 1, 10), Square(1, 1, 20)]])
    item = s1.item(0)
    item.set_parent("nuclei", 2)
    s1.update_item(item)
    assert {None, 2}.issubset(s1.df[s1.parent_id_field].unique())
    assert {None, "nuclei"}.issubset(s1.df[s1.parent_entity_field].unique())
    assert s1.item(0).df[s1.parent_entity_field].iloc[0] == "nuclei"


def test_set_entity() -> None:
    s1 = from_shapes_3d([[Square(0, 0, 10), Square(5, 5, 10)], [Square(1, 1, 10), Square(1, 1, 20)]])
    item = s1.item(0)
    item.set_entity_type("nuclei")
    s1.update_item(item)
    assert {"cell", "nuclei"}.issubset(s1.df[s1.entity_name_field].unique())
    assert s1.item(0).df[s1.entity_name_field].iloc[0] == "nuclei"


def test_empty_segmentation() -> None:
    s1 = from_shapes_3d([])
    s2 = from_shapes_3d([[Square(1, 1, 10)]], cids=[2300002])
    new_id = s1.add_item(s2.item(2300002))
    assert new_id == 2300000
