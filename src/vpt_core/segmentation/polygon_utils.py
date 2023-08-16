from dataclasses import dataclass
from typing import Dict

import numpy as np
from shapely import geometry

from vpt_core import log
from vpt_core.segmentation.geometry_utils import (
    make_polygons_from_label_matrix,
    smooth_and_simplify,
    convert_to_multipoly,
    get_valid_geometry,
)
from vpt_core.segmentation.seg_result import SegmentationResult


@dataclass(frozen=True)
class PolygonCreationParameters:
    simplification_tol: int
    smoothing_radius: int
    minimum_final_area: int


def generate_polygons_from_mask(mask: np.ndarray, polygon_parameters: Dict) -> SegmentationResult:
    log.info("generate_polygons_from_mask")
    parameters = PolygonCreationParameters(**polygon_parameters)
    seg_result = get_polygons_from_mask(mask, parameters.smoothing_radius, parameters.simplification_tol)
    seg_result.remove_polys(lambda poly: poly.area < parameters.minimum_final_area)
    return seg_result


def get_polygons_from_mask(mask: np.ndarray, smoothing_radius, simplification_tolerance) -> SegmentationResult:
    """
    Accepts either a 2D or 3D numpy array label matrix, returns a SegmentationResult of
    MultiPolygons surrounding each label/mask. Performs smoothing and simplification of
    Polygons before returning.
    """

    # If passed 2D data, expand axes so that z-level indexing will work
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=0)

    polys_data = []
    for z in range(mask.shape[0]):
        list_of_mask_ids = np.unique(mask[z, :, :])
        log.info(f"get_polygons_from_mask: z={z}, labels:{len(list_of_mask_ids)}")
        for idx, mask_id in enumerate(list_of_mask_ids):
            # Value of zero is background, skip
            if mask_id == 0:
                continue

            try:
                # Convert each region of the entity into a polygon
                raw_polys = make_polygons_from_label_matrix(mask_id, mask[z, :, :])

                polys = [
                    smooth_and_simplify(raw_poly, smoothing_radius, simplification_tolerance) for raw_poly in raw_polys
                ]
                polys = [poly for poly in polys if not poly.is_empty]

                # If smoothing and simplifying eliminated all polygons, don't add them to the output
                if len(polys) == 0:
                    continue

                # Transform the list of 1 or more mask polygons into a multipolygon
                multi_poly = convert_to_multipoly(get_valid_geometry(geometry.MultiPolygon(polys)))
                polys_data.append(
                    {
                        SegmentationResult.detection_id_field: idx + 1,
                        SegmentationResult.cell_id_field: mask_id,
                        SegmentationResult.z_index_field: z,
                        SegmentationResult.geometry_field: multi_poly,
                    }
                )
            except ValueError:
                # If the MultiPolygon is not created properly, it is probably because the
                # geometry is low quality or otherwise strange. In that situation, it's ok
                # to discard the geometry by catching the exception and not appending the
                # geometry to the output.
                log.info(f"Mask id {mask_id} could not be converted to a polygon.")

    return SegmentationResult(list_data=polys_data)


def get_upscale_matrix(scale_x: float, scale_y: float) -> np.ndarray:
    return np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
