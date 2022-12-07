from abc import ABC, abstractmethod

import numpy as np


class AbstractArm(ABC):
    @abstractmethod
    def sample(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def set(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError


class AbstractBandit(ABC):
    def __init__(self, k: int, name: str, seed: int):
        self.name = name
        self.k = k
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def sample(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def set(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError