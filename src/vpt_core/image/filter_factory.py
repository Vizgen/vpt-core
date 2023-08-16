from typing import Callable, Dict, List

from vpt_core.image import Filter, Header
from vpt_core.image.filter import (
    apply_sequence,
    create_blur_filter,
    create_downsample_filter,
    create_merlin_watershed_filter,
    create_normalization_filter,
)

factory_map: Dict[str, Callable[[dict], Filter]] = {
    "normalize": create_normalization_filter,
    "blur": create_blur_filter,
    "downsample": create_downsample_filter,
    "merlin-ws": create_merlin_watershed_filter,
}


def create_filter(h: Header) -> Filter:
    if h.name in factory_map:
        return factory_map[h.name](h.parameters)
    else:
        raise NameError(f"invalid filter name {h.name}")


def create_filter_by_sequence(headers: List[Header]) -> Filter:
    filters = [create_filter(h) for h in headers] if headers else []
    return lambda image: apply_sequence(image, filters)
