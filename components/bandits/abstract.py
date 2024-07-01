from abc import ABC, abstractmethod

import numpy as np


class AbstractArm(ABC):
    @abstractmethod
    def sample(self, *args, **kwargs):
        """
        Compute the index (e.g., ucb) of an arm
        """
        raise NotImplementedError

    @abstractmethod
    def set(self, *args, **kwargs):
        """
        Use this method to set parameters after initialization
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Update the arm
        """
        raise NotImplementedError


class AbstractBandit(ABC):
    def __init__(self, k: int, name: str, seed: int):
        self.name = name
        self.k = k
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def sample(self, *args, **kwargs):
        """
        Get the arm with the highest index
        :return: an integer
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, *args, **kwargs):
        """Update the algorithm with information from the environment"""
        raise NotImplementedError

    @abstractmethod
    def set(self, *args, **kwargs):
        """Update the parameters of the algorithm after initialization"""
        raise NotImplementedError

    def __len__(self):
        """The number of arms"""
        return self.k
