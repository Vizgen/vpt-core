import cv2
import numpy as np
from shapely import geometry, make_valid, unary_union

from vpt_core import log


def smooth_and_simplify(poly, radius, tol):
    if isinstance(poly, geometry.MultiPolygon):
        buffered_shapes = (p.buffer(-radius).buffer(radius * 2).buffer(-radius) for p in poly.geoms)
        buffered_multipolygons = (
            p if type(p) is geometry.MultiPolygon else geometry.MultiPolygon([p]) for p in buffered_shapes
        )
        buffered_polygons = (p for mp in buffered_multipolygons for p in mp.geoms)
        poly = geometry.MultiPolygon(buffered_polygons)
    elif isinstance(poly, geometry.Polygon):
        poly = poly.buffer(-radius).buffer(radius * 2).buffer(-radius)
    return largest_geometry(poly.simplify(tolerance=tol))


def largest_geometry(shape):
    """
    If passed a Polygon, returns the Polygon.
    If passed a MultiPolygon, returns the largest Polygon region.
    Else throws TypeError
    """
    if type(shape) is geometry.Polygon:
        return shape
    elif type(shape) is geometry.MultiPolygon:
        if len(shape.geoms) == 0:
            log.warning("Empty multipolygon passed, an empty polygon will be returned")
            return geometry.Polygon()
        sizes = np.array([(idx, g.area) for idx, g in enumerate(shape.geoms)])
        idx_largest = sizes[sizes[:, 1].argmax(), 0]
        return shape.geoms[int(idx_largest)]
    else:
        raise TypeError(f"Objects of type {type(shape)} are not supported")


def convert_to_multipoly(shape):
    if type(shape) is geometry.Polygon:
        return geometry.multipolygon.MultiPolygon([shape])
    elif type(shape) is geometry.MultiPolygon:
        return shape
    elif type(shape) is geometry.GeometryCollection:
        poly_shapes = [g for g in shape.geoms if type(g) in [geometry.Polygon, geometry.MultiPolygon]]
        poly = get_valid_geometry(unary_union(poly_shapes))
        return poly if type(poly) is geometry.MultiPolygon else geometry.multipolygon.MultiPolygon([poly])
    else:
        # If type is not Polygon or Multipolygon, the shape
        # is strange / small and should be rejected
        return geometry.MultiPolygon()


def make_polygons_from_label_matrix(entity_id, label_matrix):
    """
    Creates a list of Polygon objects from a label matrix
    """
    contours, _ = cv2.findContours(
        (label_matrix == entity_id).astype("uint8"),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    polys = []
    for c in contours:
        if c.shape[0] < 4:
            continue

        # Create a valid polygon object
        p = geometry.Polygon(c[:, 0, :]).buffer(0)
        p = largest_geometry(p)
        if p.is_empty:
            continue
        polys.append(p)

    return polys


def get_valid_geometry(shape):
    try:
        return make_valid(shape)
    except ValueError:
        return geometry.multipolygon.MultiPolygon()
