from typing import Optional, Dict, List, Iterable, Union

import pandas as pd

from vpt_core.segmentation.segmentation_base import SegmentationBase
from vpt_core.segmentation.seg_result import SegmentationResult
from vpt_core.io.image import ImageSet


class SegmentationBaseTest(SegmentationBase):
    @staticmethod
    def run_segmentation(
        segmentation_properties: Dict,
        segmentation_parameters: Dict,
        polygon_parameters: Dict,
        result: List[str],
        images: Optional[ImageSet] = None,
        transcripts: Optional[pd.DataFrame] = None,
    ) -> Union[SegmentationResult, Iterable[SegmentationResult]]:
        return SegmentationResult()

    @staticmethod
    def validate_task(task: Dict) -> Dict:
        return task


def test_segmentation_instance():
    x = SegmentationBaseTest()

    seg_result = x.run_segmentation({}, {}, {}, ["cell"])
    assert type(seg_result) is SegmentationResult

    task = x.validate_task({"foo": "bar"})
    assert task.get("foo") == "bar"
