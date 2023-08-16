from dataclasses import dataclass
from typing import List

from vpt_core.image import Header


@dataclass
class InputData:
    image_channel: str
    image_preprocessing: List[Header]

    def __init__(self, image_channel: str, image_preprocessing: List[dict]):
        self.image_channel = image_channel
        self.image_preprocessing = [Header(**x) for x in image_preprocessing]
