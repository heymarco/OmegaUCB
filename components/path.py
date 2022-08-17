from typing import Tuple, List

import numpy as np
import pandas as pd


class PathElement:
    def __init__(self,
                 data: Tuple[np.ndarray, np.ndarray],
                 time: int):
        self.data = data
        self.time = time

    def budget(self) -> int:
        return len(self.data[-1]) * self.time


class Path:
    def __init__(self,
                 elements: List[PathElement] = []):
        self.elements: List[PathElement] = elements

    def __len__(self):
        return len(self.elements)

    def time(self) -> int:
        if len(self.elements) == 0:
            return 0
        else:
            return int(np.sum([element.time for element in self.elements]))

    def budget(self) -> int:
        if len(self.elements) == 0:
            return 0
        else:
            return int(np.sum([element.budget() for element in self.elements]))

    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        chunks = [e.data for e in self.elements]
        x = [c[0] for c in chunks]
        y = [c[1] for c in chunks]
        x = np.vstack(x)
        y = np.vstack(y).flatten()
        return x, y

    def vector(self) -> np.array:
        x = self.time()
        y = np.sum([
            len(e.data[-1]) for e in self.elements
        ])
        return np.array([x, y])

    def subpath_without_index(self, index: int):
        assert index < len(self)
        subpath_elements = [self.elements[i] for i in range(len(self)) if i != index]
        subpath = Path(subpath_elements)
        return subpath

