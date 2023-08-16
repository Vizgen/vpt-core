from __future__ import annotations

from dataclasses import dataclass
from typing import Set, Dict, Callable, Optional

from shapely.geometry.base import BaseGeometry

import numpy as np
from geopandas import GeoDataFrame
from vpt_core.segmentation.geometry_utils import convert_to_multipoly, get_valid_geometry

from vpt_core.segmentation import ResultFields


@dataclass
class SegmentationItem(ResultFields):
    def __init__(self, entity_type: str, df: GeoDataFrame):
        self.entity_type = entity_type
        self.df = df

    def get_entity_id(self) -> int:
        return int(self.df[self.cell_id_field].iloc[0])

    def get_entity_type(self) -> str:
        return self.entity_type

    def set_entity_type(self, entity_type: str) -> None:
        self.entity_type = entity_type
        self.df = self.df.assign(**{self.entity_name_field: entity_type})

    def set_parent(self, parent_entity_type: Optional[str], parent_entity_id: Optional[int]) -> None:
        self.df = self.df.assign(
            **{self.parent_id_field: parent_entity_id, self.parent_entity_field: parent_entity_type}
        )
        self.df.loc[:, self.parent_id_field] = self.df.loc[:, self.parent_id_field].astype("object")

    def get_volume(self):
        return np.sum([x.area for x in self.df[self.geometry_field]])

    def get_z_planes(self) -> Set[int]:
        return set(self.df[self.z_index_field])

    def as_geometries_by_z(self) -> Dict:
        return dict(
            zip(
                self.df[self.z_index_field],
                self.df[self.geometry_field],
            )
        )

    def as_copy(self) -> SegmentationItem:
        return SegmentationItem(self.entity_type, self.df.copy())

    def get_overlapping_volume(self, item: SegmentationItem):
        """
        Calulates intersection area for 3D shapes stored in a geo data frame.
        """
        left_entity = self.as_geometries_by_z()
        right_entity = item.as_geometries_by_z()
        intersection_area = 0
        for z in left_entity:
            if z in right_entity:
                intersection_area += (
                    get_valid_geometry(left_entity[z]).intersection(get_valid_geometry(right_entity[z])).area
                )
        return intersection_area

    @staticmethod
    def process_same_plane_geometries(
        target: GeoDataFrame, arg: GeoDataFrame, op: Callable[[BaseGeometry, BaseGeometry], BaseGeometry]
    ) -> None:
        z_planes_in_both = set(target[SegmentationItem.z_index_field]) & set(arg[SegmentationItem.z_index_field])

        def get_geometry(df: GeoDataFrame, z: int):
            idx = df.loc[df[SegmentationItem.z_index_field] == z].index[0]
            item = df.at[idx, SegmentationItem.geometry_field]
            return idx, get_valid_geometry(item)

        for z in z_planes_in_both:
            # Trims the larger geometry with a small buffer
            ind, valid_item = get_geometry(target, z)

            _, valid_arg = get_geometry(arg, z)

            result_raw = op(valid_item, valid_arg)
            result_geometry = convert_to_multipoly(get_valid_geometry(result_raw))

            if not result_geometry.is_empty:
                # Overwrites large geometry with trimmed geometry
                target.at[ind, SegmentationItem.geometry_field] = result_geometry
            else:
                target.drop(ind, inplace=True)


def intersection(a: SegmentationItem, b: SegmentationItem) -> SegmentationItem:
    result = a.as_copy()
    z_field = SegmentationItem.z_index_field
    result.df.drop(result.df[~result.df[z_field].isin(b.df[z_field])].index, inplace=True)
    SegmentationItem.process_same_plane_geometries(result.df, b.df, lambda a, b: a.intersection(b))
    return result


def difference(a: SegmentationItem, b: SegmentationItem, min_distance: float = 1) -> SegmentationItem:
    result = a.as_copy()
    SegmentationItem.process_same_plane_geometries(
        result.df, b.df, lambda a, b: a.difference(b.buffer(max(min_distance, 1e-10)))
    )
    return result
