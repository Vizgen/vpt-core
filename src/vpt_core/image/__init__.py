from dataclasses import dataclass
from typing import Callable, List, Union

import numpy as np

Filter = Callable[[Union[np.ndarray, List[np.ndarray]]], Union[np.ndarray, List[np.ndarray]]]


@dataclass
class Header:
    name: str
    parameters: dict

    def __init__(self, name: str, parameters=None):
        if parameters is None:
            parameters = {}
        self.name = name
        self.parameters = parameters
