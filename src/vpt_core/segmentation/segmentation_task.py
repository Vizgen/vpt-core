from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SegTask:
    task_id: int
    segmentation_family: str
    entity_types_detected: List[str]
    z_layers: List[int]
    task_input_data: List
    segmentation_parameters: Dict
    segmentation_properties: Dict
    polygon_parameters: Dict
