from __future__ import annotations

from collections import defaultdict, namedtuple
from typing import List, Optional, Set, Tuple, Callable, Iterable

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from shapely import affinity, geometry, MultiPolygon
from shapely.ops import unary_union

from vpt_core import log
from vpt_core.segmentation import ResultFields
from vpt_core.segmentation.geometry_utils import get_valid_geometry, convert_to_multipoly
from vpt_core.segmentation.segmentation_item import SegmentationItem


class SegmentationResult(ResultFields):
    MAX_ENTITY_ID = 99999
    MAX_TILE_ID = 99999
    MAX_TASK_ID = 9

    def __init__(
        self,
        list_data=None,
        dataframe: Optional[GeoDataFrame] = None,
        entity: Optional[str] = None,
    ):
        if list_data is None and dataframe is None:
            list_data = []
        if dataframe is not None:
            self.df = dataframe.copy()
        else:
            self.df = GeoDataFrame(
                list_data,
                columns=[
                    self.detection_id_field,
                    self.cell_id_field,
                    self.z_index_field,
                    self.geometry_field,
                ],
            )
        if entity is not None:
            self.entity_type = entity
        self.df.set_geometry(self.geometry_field, inplace=True)

        if self.parent_entity_field not in self.df.columns:
            self.set_column(self.parent_entity_field, None)
        if self.parent_id_field not in self.df.columns:
            self.set_column(self.parent_id_field, None)

        self.df.loc[:, self.cell_id_field] = self.df.loc[:, self.cell_id_field].astype("int64")

    def item(self, item_id: int) -> SegmentationItem:
        df = self.df.loc[self.df[self.cell_id_field] == item_id]
        if len(df) < 1:
            raise KeyError(f"invalid item id {item_id}")

        return SegmentationItem(self.entity_type if hasattr(self, "entity_type") else "", df)

    @staticmethod
    def get_connected_items(processed_item: SegmentationItem) -> List[SegmentationItem]:
        data = SegmentationResult._separate_multi_geometries(processed_item.df)
        seg_res = SegmentationResult(dataframe=data)
        seg_res.fuse_across_z(coverage_threshold=-0.1)
        items = []
        for cell_id in seg_res.df[processed_item.cell_id_field].unique():
            item = seg_res.item(cell_id)
            item.set_entity_type(processed_item.get_entity_type())
            item.df = item.df.assign(**{processed_item.cell_id_field: processed_item.get_entity_id()})
            items.append(item)
        return items

    def update_item(self, item: SegmentationItem):
        clone = item.as_copy()
        self.df = pd.concat(
            [self.df.drop(self.df.loc[self.df[self.cell_id_field] == item.get_entity_id()].index), clone.df],
            ignore_index=True,
        )

    def add_item(self, item: SegmentationItem) -> int:
        clone = item.as_copy()
        clone.set_entity_type(self.entity_type)
        # todo implement new id generation
        if len(self.df) > 0:
            new_id = self.df[SegmentationItem.cell_id_field].max() + 1
        else:
            entity_id_len = len(str(SegmentationResult.MAX_ENTITY_ID))
            id_prefix = str(item.get_entity_id())[:-entity_id_len]
            new_id = np.int64(id_prefix + "0".zfill(entity_id_len))
        clone.df[SegmentationItem.cell_id_field] = clone.df[SegmentationItem.cell_id_field].apply(lambda x: new_id)
        self.df = pd.concat([self.df, clone.df], ignore_index=True)
        return new_id

    def update_column(self, column_name: str, callback: Callable, *args, **kwargs):
        self.df[column_name] = self.df[column_name].apply(lambda x: callback(x, *args, **kwargs), convert_dtype=False)

    def update_geometry(self, callback: Callable, *args, **kwargs):
        def transform(x) -> geometry.MultiPolygon:
            return convert_to_multipoly(callback(x, *args, **kwargs))

        self.df[self.geometry_field] = self.df[self.geometry_field].apply(transform)

    def set_column(self, column_name: str, data):
        kwargs = {column_name: data}
        self.df = self.df.assign(**kwargs)

    def remove_column(self, column_name):
        if column_name in self.df.columns:
            self.df.drop(columns=[column_name], inplace=True)

    def set_entity_type(self, entity: str):
        self.entity_type = entity

    def translate_geoms(self, dx, dy):
        self.update_geometry(affinity.translate, xoff=dx, yoff=dy)

    def transform_geoms(self, matrix):
        tform_flat = [*matrix[:2, :2].flatten(), *matrix[:2, 2].flatten()]
        self.update_geometry(affinity.affine_transform, matrix=tform_flat)

    def remove_polys(self, condition, *args, **kwargs):
        indices_to_remove = []
        for idx, row in self.df.iterrows():
            if condition(row[self.geometry_field], *args, **kwargs):
                indices_to_remove.append(idx)
        if len(indices_to_remove) > 0:
            self.df.drop(indices_to_remove, inplace=True)

    def _get_replication_pairs(self, z_planes: Set[int]):
        copy_pairs: List[Tuple[int, int]] = []
        if len(self.df) < 1:
            return copy_pairs
        occupied = set(self.df["ZIndex"].to_list())
        exceptional_z = min(z_planes) - 1
        while z_planes != occupied:
            missing_layers = z_planes.difference(occupied)
            neighbor_layers = [
                [present if abs(present - missing) == 1 else exceptional_z for present in occupied]
                for missing in missing_layers
            ]
            layer_pairs = [
                (missing, max(neighbor))
                for missing, neighbor in zip(missing_layers, neighbor_layers)
                if any([z != exceptional_z for z in neighbor])
            ]
            if len(layer_pairs) == 0:
                return copy_pairs

            copy_pairs.extend(layer_pairs)
            for x in layer_pairs:
                occupied.add(x[0])
        return copy_pairs

    def replicate_across_z(self, z_planes: List[int]):
        replicate_pairs = self._get_replication_pairs(set(z_planes))
        for missing_z, copy_z in replicate_pairs:
            missing_z_seg = SegmentationResult(dataframe=self.df.loc[self.df[self.z_index_field] == copy_z])
            missing_z_seg.set_column(self.z_index_field, missing_z)
            self.df = self.combine_segmentations([self, missing_z_seg]).df

    def set_z_levels(self, z_levels: List[float], column_name):
        levels = [z_levels[row[self.z_index_field]] for _, row in self.df.iterrows()]
        self.set_column(column_name, levels)

    def _assign_entity_index(
        self, entities_df: GeoDataFrame, entities_ids: List, cur_poly: MultiPolygon, coverage_threshold: float
    ) -> List[int]:
        res = []
        id_values = entities_df[self.cell_id_field].values
        geoms = entities_df.geometry.values
        for i in entities_ids:
            prev_poly = geoms[id_values == i][0]
            polys_intersection = prev_poly.intersection(cur_poly)
            if (polys_intersection.area / min(prev_poly.area, cur_poly.area)) > coverage_threshold:
                res.append(i)
        return res

    def _update_entities_index(
        self, overlaps_pairs: np.ndarray, cur_cells_ids: np.ndarray, cur_df: GeoDataFrame, updated_ids: List
    ):
        id_values = cur_df[self.cell_id_field].values
        index_values = cur_df.index
        for old_id, new_id in updated_ids:
            for j in index_values[np.where(id_values == old_id)]:
                self.df.at[j, self.cell_id_field] = new_id
                cur_df.at[j, self.cell_id_field] = new_id
            cur_cells_ids[cur_cells_ids == old_id] = new_id

            if len(overlaps_pairs) == 0:
                return
            overlaps_pairs[overlaps_pairs[:, 0] == old_id, 0] = new_id

    def fuse_across_z(self, coverage_threshold: float = 0.5):
        if len(self.df) == 0:
            return
        z_lines = self.df["ZIndex"].unique()
        reserved_ids = []
        prev_df = self.df.loc[self.df[self.z_index_field] == z_lines[0]]
        for z_i in range(1, len(z_lines)):
            prev_df = prev_df.assign(**{self.z_index_field: z_lines[z_i]})
            cur_df = self.df.loc[self.df[self.z_index_field] == z_lines[z_i]]

            reserved_ids.extend(prev_df[self.cell_id_field].to_list())
            cell_ids_values = cur_df[self.cell_id_field].values
            cur_cells_ids = np.unique(cell_ids_values)
            cur_geoms = cur_df.geometry.values
            overlaps = np.array(self.find_overlapping_entities(prev_df, cur_df))

            for i in range(len(cur_cells_ids)):
                cur_i = cur_cells_ids[i]
                cur_intersected_i_to_update = []
                need_new_index = True
                if len(overlaps) > 0 and cur_i in overlaps[:, 0]:
                    cur_poly = cur_geoms[cell_ids_values == cur_i][0]
                    prev_ids = [pair[1] for pair in overlaps if pair[0] == cur_i]

                    prev_ids = self._assign_entity_index(prev_df, prev_ids, cur_poly, coverage_threshold)
                    if len(prev_ids) > 1:
                        prev_i_to_update = [(prev_i, prev_ids[0]) for prev_i in prev_ids[1:]]
                        for z in z_lines[:z_i]:
                            df_to_update = self.df.loc[self.df[self.z_index_field] == z]
                            self._update_entities_index(np.array([]), np.array([]), df_to_update, prev_i_to_update)

                    if len(prev_ids) > 0:
                        prev_i = prev_ids[0]
                        need_new_index = False
                        if prev_i != cur_i:
                            if prev_i in cur_cells_ids:
                                new_id = max(max(reserved_ids), max(cur_cells_ids)) + 1
                                cur_intersected_i_to_update.append((prev_i, new_id))
                            cur_intersected_i_to_update.append((cur_i, prev_i))

                if need_new_index:
                    if cur_i in reserved_ids:
                        new_id = max(max(reserved_ids), max(cur_cells_ids)) + 1
                        cur_intersected_i_to_update.append((cur_i, new_id))
                self._update_entities_index(overlaps, cur_cells_ids, cur_df, cur_intersected_i_to_update)

            prev_df = cur_df

        self.group_duplicated_entities()

    def group_duplicated_entities(self):
        duplicated_fields = [self.z_index_field, self.cell_id_field]
        duplicated = self.df.duplicated(duplicated_fields, keep=False)
        if len(duplicated) == 0:
            return

        grouped = (
            self.df[duplicated]
            .groupby(duplicated_fields)[self.geometry_field]
            .apply(lambda geoms: convert_to_multipoly(get_valid_geometry(unary_union(geoms))))
        )

        self.df.drop_duplicates(duplicated_fields, inplace=True)
        merged_geoms = self.df[duplicated].apply(
            lambda row: grouped[row[self.z_index_field], row[self.cell_id_field]], axis=1
        )
        if len(merged_geoms) > 0:
            self.df[self.geometry_field].update(merged_geoms)

    @staticmethod
    def _separate_multi_geometries(data: GeoDataFrame) -> GeoDataFrame:
        geoms_separated = data.copy()
        geometry_field = SegmentationResult.geometry_field
        new_rows = []
        for index, row in geoms_separated.iterrows():
            i = 0
            for geom in row[geometry_field].geoms:
                if i == 0:
                    i += 1
                    geoms_separated.at[index, geometry_field] = convert_to_multipoly(get_valid_geometry(geom))
                    continue
                new_row = geoms_separated[geoms_separated.index == index].copy()
                new_row[geometry_field] = convert_to_multipoly(get_valid_geometry(geom))
                new_rows.append(new_row)
        geoms_separated = pd.concat([geoms_separated, *new_rows], ignore_index=True)
        geoms_separated = geoms_separated.assign(**{SegmentationResult.cell_id_field: range(len(geoms_separated))})
        return geoms_separated

    def remove_edge_polys(self, tile_size: Tuple[int, int]):
        crop_frame = geometry.box(-10, -10, tile_size[0] + 10, tile_size[1] + 10) - geometry.box(
            5, 5, tile_size[0] - 5, tile_size[1] - 5
        )
        for entity_id in self.df[self.cell_id_field].unique():
            cur_gdf = self.df.loc[self.df[self.cell_id_field] == entity_id]
            is_intersected = False
            for z in cur_gdf[self.z_index_field]:
                cur_geom_df = cur_gdf.loc[cur_gdf[self.z_index_field] == z, self.geometry_field]
                if len(cur_geom_df) == 0:
                    continue
                if any(cur_geom_df.values.intersects(crop_frame)):
                    is_intersected = True
                    break
            if is_intersected:
                self.df = self.df.drop(self.df.loc[self.df[self.cell_id_field] == entity_id].index)

    def _resolve_cell_overlap(self, large_cell: GeoDataFrame, small_cell: GeoDataFrame, min_distance: int):
        """
        Trims area from larger entity
        """
        z_planes_in_both = set(large_cell[self.z_index_field]).intersection(set(small_cell[self.z_index_field]))

        for z in z_planes_in_both:
            # Gets the large and small geometries
            large_at_z_idx = large_cell.loc[large_cell[self.z_index_field] == z].index[0]
            small_at_z_idx = small_cell.loc[small_cell[self.z_index_field] == z].index[0]

            large_at_z = self.df.at[large_at_z_idx, self.geometry_field]
            small_at_z = self.df.at[small_at_z_idx, self.geometry_field]

            try:
                # Trims the larger geometry with a small buffer
                valid_large = get_valid_geometry(large_at_z)
                valid_small = get_valid_geometry(small_at_z)

                trimmed_raw = valid_large.difference(valid_small.buffer(min_distance))
                trimmed_geometry = convert_to_multipoly(get_valid_geometry(trimmed_raw))
            except ValueError:
                log.info("Entity could not be converted to a valid polygon, removing.")
                trimmed_geometry = geometry.MultiPolygon()

            # Overwrites large geometry with trimmed geometry
            self.df.at[large_at_z_idx, self.geometry_field] = trimmed_geometry

    @staticmethod
    def find_overlapping_entities(dataframe_1: GeoDataFrame, dataframe_2: Optional[GeoDataFrame] = None) -> List:
        """
        Uses shapely library to rapidly identify entities that overlap in 3D
        """

        gdf2 = dataframe_2 if dataframe_2 is not None else dataframe_1

        overlaps = dataframe_1.sindex.query(gdf2.geometry, predicate="intersects").T

        # Remove self-intersections
        if dataframe_2 is None:
            overlaps = np.array([pair for pair in overlaps if pair[0] != pair[1]])
        if len(overlaps) == 0:
            return []

        z_field = SegmentationResult.z_index_field
        id_field = SegmentationResult.cell_id_field

        same_z_overlaps = gdf2[z_field].values[overlaps[:, 0]] == dataframe_1[z_field].values[overlaps[:, 1]]

        # Map list positions back to Entity IDs
        entity_overlap = np.transpose(
            [gdf2[id_field].values[overlaps[:, 0]], dataframe_1[id_field].values[overlaps[:, 1]]]
        )[same_z_overlaps]

        res: List = []
        # Remove repeated sets of EntityIDs
        if dataframe_2 is None:
            res = list(set(frozenset(pair) for pair in entity_overlap))
        else:
            res = list(set(tuple(pair) for pair in entity_overlap))
        return res

    def get_volume(self):
        return np.sum([x.area for x in self.df[self.geometry_field]])

    def get_z_planes(self) -> Set:
        return set(self.df[self.z_index_field])

    def get_z_geoms(self, z_line: int) -> GeoSeries:
        return self.df.loc[self.df[self.z_index_field] == z_line, self.geometry_field]

    @staticmethod
    def combine_segmentations(segmentations: List[SegmentationResult]) -> SegmentationResult:
        non_empty_segmentations = [seg for seg in segmentations if len(seg.df) > 0]
        if len(non_empty_segmentations) > 1:
            to_concat = [seg.df for seg in non_empty_segmentations]
            df = GeoDataFrame(pd.concat(to_concat, ignore_index=True))
            duplicated_fields = [
                SegmentationResult.cell_id_field,
                SegmentationResult.z_index_field,
            ]
            deprecated_indexes = df[df.duplicated(duplicated_fields)].index
            if len(deprecated_indexes) > 0:
                log.warning(
                    f"Found several entity rows for the same z planes. {len(deprecated_indexes)} rows will be "
                    f"removed from segmentation result."
                )
                df.drop(deprecated_indexes, inplace=True)
            res = SegmentationResult(dataframe=df)
            res.set_column(SegmentationResult.detection_id_field, list(range(len(res.df))))
            return res
        elif len(non_empty_segmentations) == 1:
            return non_empty_segmentations[0]
        else:
            return segmentations[0] if len(segmentations) > 0 else SegmentationResult()

    def _union_entities(self, base_gdf, add_gdf):
        """
        Adds the area from entity_id_add to entity_id_base across z-levels
        """

        # Area of intersection
        add_entity = dict(zip(add_gdf[self.z_index_field], add_gdf[self.geometry_field]))

        occupied_z_planes = set(base_gdf[self.z_index_field]).intersection(set(add_entity))
        for z in occupied_z_planes:
            base_geoms = base_gdf.loc[base_gdf[self.z_index_field] == z]
            if len(base_geoms) == 0:
                continue
            if z not in add_entity:
                continue
            base_geom_z_idx = base_geoms.index[0]

            base_geom = base_gdf.at[base_geom_z_idx, self.geometry_field]
            add_geom = add_entity[z]

            try:
                valid_base = get_valid_geometry(base_geom)
                valid_add = get_valid_geometry(add_geom)

                union_raw = valid_base.union(valid_add)
                union_geom = convert_to_multipoly(get_valid_geometry(union_raw))
            except ValueError:
                log.info("Entity could not be converted to a valid polygon, removing.")
                union_geom = geometry.MultiPolygon()

            # Overwrites large geometry with trimmed geometry
            self.df.at[base_geom_z_idx, self.geometry_field] = union_geom

    def cell_size_filter(self, minimum_area: int):
        depricated_entity_ids = []
        for entity_id, gdf in self.df.groupby(self.cell_id_field):
            if len(gdf) > 0 and not any(cell.area > minimum_area for cell in gdf[self.geometry_field]):
                depricated_entity_ids.append(entity_id)
        self._drop_by_entity_id(depricated_entity_ids)

    def make_non_overlapping_polys(self, min_distance: int = 2, min_area: int = 100, log_progress: bool = False):
        # Find cells that have any overlapping area
        problem_sets = self.find_overlapping_entities(self.df)
        log.info(f"Found {len(problem_sets)} overlaps")

        # For each pair of overlapping cells, resolve the conflict
        depricated_entity_ids = []

        def iterate(x: Iterable):
            if log_progress:
                return log.show_progress(x)
            else:
                return x

        # Union the large overlap Entities
        for problem in iterate(problem_sets):
            # Get the slice of the dataframe for each entity
            pl = list(problem)
            entity_id_left, entity_id_right = pl[0], pl[1]

            # If either cell is in the depricated id list, ignore the overlap.
            # The entity union / drop process may drop a cell that should be retained.
            if entity_id_left in depricated_entity_ids or entity_id_right in depricated_entity_ids:
                continue
            left = self.item(entity_id_left)
            right = self.item(entity_id_right)

            # Find seperate and overlapping volumes of the cells
            volume_left = left.get_volume()
            volume_right = right.get_volume()
            overlap_volume = left.get_overlapping_volume(right)
            overlap_volume_percent = overlap_volume / min(volume_right, volume_left)

            # If overlap is > 50% of either cell, eliminate the small cell and keep the big one
            if overlap_volume_percent > 0.5:
                if volume_left > volume_right:
                    self._union_entities(left.df, right.df)
                    depricated_entity_ids.append(entity_id_right)
                else:
                    self._union_entities(right.df, left.df)
                    depricated_entity_ids.append(entity_id_left)

        # With large overlaps resolved, re-identify problem sets and trim overlaps
        self._drop_by_entity_id(depricated_entity_ids)
        problem_sets = self.find_overlapping_entities(self.df)
        log.info(f"After union of large overlaps, found {len(problem_sets)} overlaps")

        for problem in iterate(problem_sets):
            # Get the slice of the dataframe for each entity
            pl = list(problem)
            entity_id_left, entity_id_right = pl[0], pl[1]
            left = self.item(entity_id_left)
            right = self.item(entity_id_right)

            # Find seperate and overlapping volumes of the cells
            volume_left = left.get_volume()
            volume_right = right.get_volume()

            # Trim the larger cell to dodge the smaller cell
            if volume_left > volume_right:
                self._resolve_cell_overlap(left.df, right.df, min_distance)
            else:
                self._resolve_cell_overlap(right.df, left.df, min_distance)

        # After both steps, check for any remaining overlaps
        problem_sets = self.find_overlapping_entities(self.df)
        log.info(f"After both resolution steps, found {len(problem_sets)} uncaught overlaps")

        # Filter any small cells that were created
        self.cell_size_filter(min_area)

    def union_intersections(self, min_distance: int, min_area: int):
        problem_sets = self.find_overlapping_entities(self.df)
        depricated_entity_ids = []
        for problem in problem_sets:
            entity_id_left, entity_id_right, *_ = list(problem)
            left_df = self.df.loc[self.df[self.cell_id_field] == entity_id_left]
            right_df = self.df.loc[self.df[self.cell_id_field] == entity_id_right]
            self._union_entities(left_df, right_df)
            depricated_entity_ids.append(entity_id_right)
        self._drop_by_entity_id(depricated_entity_ids)
        self.cell_size_filter(min_area)

    def larger_resolve_intersections(self, min_distance: int, min_area: int):
        problem_sets = self.find_overlapping_entities(self.df)
        depricated_entity_ids = []
        for problem in problem_sets:
            entity_id_left, entity_id_right, *_ = list(problem)
            left = self.item(entity_id_left)
            right = self.item(entity_id_right)
            left_area, right_area = left.get_volume(), right.get_volume()
            if left_area > right_area:
                depricated_entity_ids.append(entity_id_right)
            else:
                depricated_entity_ids.append(entity_id_left)
        self._drop_by_entity_id(depricated_entity_ids)
        self.cell_size_filter(min_area)

    def _drop_by_entity_id(self, entity_ids_to_delete):
        depricated_row_indexes = []
        for entity_id in entity_ids_to_delete:
            row_indexes = self.df.loc[self.df[self.cell_id_field] == entity_id].index.to_list()
            depricated_row_indexes.extend(row_indexes)
        self.df.drop(depricated_row_indexes, inplace=True)

    @staticmethod
    def reindex_by_task(tasks_results: List[SegmentationResult], tasks_numbers: List[int]):
        def add_task_to_entity_id(old_id, task_number) -> np.int64:
            entity_str_len = len(str(SegmentationResult.MAX_ENTITY_ID))
            task_str_len = len(str(SegmentationResult.MAX_TASK_ID))

            old_id_str = str(old_id).zfill(entity_str_len)
            task_str = str(task_number + 1).zfill(task_str_len)

            if len(old_id_str) > entity_str_len or len(task_str) > task_str_len:
                raise OverflowError("EntityID is out of the int64 type range")
            return np.int64("".join([task_str, old_id_str]))

        for task_i in range(len(tasks_results)):
            tasks_results[task_i].update_column(
                SegmentationResult.cell_id_field,
                add_task_to_entity_id,
                task_number=tasks_numbers[task_i],
            )
        return tasks_results

    def create_relationships(self, parent_seg: SegmentationResult, coverage_threshold: float):
        if len(parent_seg.df) == 0 or len(self.df) == 0:
            return
        problem_sets = self.find_overlapping_entities(self.df, parent_seg.df)

        # collecting possible relations for each child entity
        relationships = defaultdict(list)
        RelationInfo = namedtuple("RelationInfo", "parent_id child_rate parent_rate overlap")
        for problem in problem_sets:
            pl = list(problem)
            parent_id, child_id = pl[0], pl[1]

            child = self.item(child_id)
            parent_entity = parent_seg.item(parent_id)

            overlapping = parent_entity.get_overlapping_volume(child)
            parent_rate = overlapping / parent_entity.get_volume()
            child_rate = overlapping / child.get_volume()
            if max(child_rate, parent_rate) > coverage_threshold:
                relationships[child_id].append(RelationInfo(parent_id, child_rate, parent_rate, overlapping))

        # get from the possible relations the child entities that are smaller than the parent entities
        result = {child_id: max(value, key=lambda x: x.child_rate) for child_id, value in relationships.items()}
        result = {
            child_id: value.parent_id for child_id, value in result.items() if value.child_rate > coverage_threshold
        }

        # get from the possible relations the child entities that are larger than the parent entities
        other = {
            child_id: max(value, key=lambda x: x.overlap)
            for child_id, value in relationships.items()
            if child_id not in result
        }
        result.update(
            {
                child_id: value.parent_id
                for child_id, value in other.items()
                if value.parent_id not in result.values() and value.parent_rate > coverage_threshold
            }
        )

        parent_ids = pd.array(
            [result.get(row[self.cell_id_field]) for _, row in self.df.iterrows()],
            dtype="object",
        )
        parent_types = [parent_seg.entity_type if parent_id is not None else None for parent_id in parent_ids]
        self.set_column(self.parent_id_field, data=parent_ids)
        self.set_column(self.parent_entity_field, data=parent_types)
