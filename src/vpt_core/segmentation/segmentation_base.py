from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Iterable, Union

import pandas as pd

from vpt_core.io.image import ImageSet
from vpt_core.segmentation.seg_result import SegmentationResult


class SegmentationBase(ABC):
    @staticmethod
    @abstractmethod
    def run_segmentation(
        segmentation_properties: Dict,
        segmentation_parameters: Dict,
        polygon_parameters: Dict,
        result: List[str],
        images: Optional[ImageSet] = None,
        transcripts: Optional[pd.DataFrame] = None,
    ) -> Union[SegmentationResult, Iterable[SegmentationResult]]:
        pass

    @staticmethod
    @abstractmethod
    def validate_task(task: Dict) -> Dict:
        pass
