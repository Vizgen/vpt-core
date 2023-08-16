import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from shapely.errors import ShapelyDeprecationWarning

import vpt_core.log as log
from vpt_core.segmentation.seg_result import SegmentationResult


@dataclass
class PolygonParams:
    min_final_area: int = 0
    min_distance_between_entities: int = 2


@dataclass
class SegFusion:
    entity_fusion_strategy: str
    fused_polygon_postprocessing_parameters: PolygonParams = PolygonParams()

    def __post_init__(self):
        if isinstance(self.fused_polygon_postprocessing_parameters, dict):
            self.fused_polygon_postprocessing_parameters = PolygonParams(**self.fused_polygon_postprocessing_parameters)


def run_harmonization(segmentations: List[SegmentationResult], min_distance: int, min_area: int):
    segmentation: SegmentationResult = SegmentationResult.combine_segmentations(segmentations)
    segmentation.make_non_overlapping_polys(min_distance, min_area)
    return segmentation


def run_union_fusion(segmentations: List[SegmentationResult], min_distance: int, min_area: int):
    segmentation: SegmentationResult = SegmentationResult.combine_segmentations(segmentations)
    segmentation.union_intersections(min_distance, min_area)
    return segmentation


def run_larger_fusion(segmentations: List[SegmentationResult], min_distance: int, min_area: int):
    segmentation: SegmentationResult = SegmentationResult.combine_segmentations(segmentations)
    segmentation.larger_resolve_intersections(min_distance, min_area)
    return segmentation


class FusionCallbacks(Enum):
    HARMONIZE = ("harmonize", run_harmonization)
    LARGER = ("larger", run_larger_fusion)
    UNION = ("union", run_union_fusion)


def fuse_task_polygons(
    segmentation_results: List[SegmentationResult],
    fusion_parameters: Dict[str, SegFusion],
):
    log.info("fuse_task_polygons")
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
    results = []
    tasks_entities = [res.entity_type for res in segmentation_results]

    for entity_type in set(tasks_entities):
        parameters = fusion_parameters[entity_type].fused_polygon_postprocessing_parameters
        strategy_key = fusion_parameters[entity_type].entity_fusion_strategy.upper()
        if strategy_key not in FusionCallbacks.__members__:
            raise Exception("Invalid fusion strategy")

        cur_results = [seg_res for i, seg_res in enumerate(segmentation_results) if tasks_entities[i] == entity_type]
        seg_result = FusionCallbacks[strategy_key].value[1](
            cur_results,
            parameters.min_distance_between_entities,
            parameters.min_final_area,
        )
        seg_result.set_entity_type(entity_type)
        results.append(seg_result)

    return results
