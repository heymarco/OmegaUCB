from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np


class Oracle:
    def __init__(self, y: np.ndarray, seed: int):
        self.y = y
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.queried_indices = []
        self.n_classes = len(np.unique(y))

    def query(self, t):
        index = self.rng.integers(low=0, high=len(self.y))
        while index in self.queried_indices:
            index = self.rng.integers(low=0, high=len(self.y))
        self.queried_indices.append(index)
        return self._create_noisy_label(self.y[index], t)

    def queried_index(self):
        return self.queried_indices[-1]

    @abstractmethod
    def _create_noisy_label(self, y, t):
        raise NotImplementedError


class SymmetricCrowdsourcingOracle(Oracle):
    def __init__(self, y: np.ndarray, p: float, seed: int):
        super(SymmetricCrowdsourcingOracle, self).__init__(y, seed)
        self.p = p
        self._classes = np.unique(self.y)

    def _create_noisy_label(self, y, t):
        t = max(int(t), 1)
        labels = []
        for _ in range(t):
            label = self._flip_label(y) if self.rng.random() < self.p else y
            labels.append(label)
        return self._mv(labels)

    def _flip_label(self, y):
        true_label = y
        y = self.rng.choice(self._classes)
        while true_label == y:
            y = self.rng.choice(self._classes)
        return y

    def _mv(self, y: np.ndarray):
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]


class LabeledPool:
    def __init__(self):
        self._pool = {}

    def x(self) -> np.ndarray:
        if len(self._pool) == 0:
            return np.array([])
        return np.array([
            self._pool[t][i][0] for t in self._pool.keys() for i in range(len(self._pool[t]))
        ])

    def y(self) -> np.ndarray:
        if len(self._pool) == 0:
            return np.array([])
        return np.array([
            self._pool[t][i][1] for t in self._pool.keys() for i in range(len(self._pool[t]))
        ])

    def x_t(self, t: float):
        if len(self._pool) == 0:
            return np.array([])
        return np.vstack([
            self._pool[t][i][0] for i in range(len(self._pool[t]))
        ])

    def y_t(self, t: float):
        if len(self._pool) == 0:
            return np.array([])
        return np.array([
            self._pool[t][i][1] for i in range(len(self._pool[t]))
        ])

    def add(self, t: float, tpl: Tuple[np.ndarray, Union[int, np.ndarray]]):
        if not t in self._pool:
            self._pool[t] = []
        self._pool[t].append(tpl)


class MajorityVotedLabeledPool(LabeledPool):
    def _mv(self, y: np.ndarray):
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]
    
    def y_t(self, t: float) -> np.ndarray:
        if len(self._pool) == 0:
            return np.array([])
        y = super(MajorityVotedLabeledPool, self).y_t(t)
        y = np.array([
            self._mv(labels) for labels in y
        ])
        return y

    def y(self) -> np.ndarray:
        y = super(MajorityVotedLabeledPool, self).y()
        y = np.array([
            self._mv(labels) for labels in y
        ])
        return y
