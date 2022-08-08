import numpy as np

from .abstract import QueryStrategy


class RandomQueryStrategy(QueryStrategy):
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def query(self, data: np.ndarray, all_indices: np.ndarray) -> int:
        return self.rng.choice(all_indices)
