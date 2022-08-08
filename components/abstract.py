from abc import ABC, abstractmethod
from typing import Tuple

from better_abc import abstract_attribute

import numpy as np

from experiment_logging import ExperimentLogger


class UsesCustomRNG(ABC):
    rng: np.random.Generator = abstract_attribute()


class QueryStrategy(UsesCustomRNG):
    @abstractmethod
    def query(self, data: np.ndarray, all_indices: np.ndarray) -> int:
        raise NotImplementedError


class Dataset(UsesCustomRNG):
    _label_dict: dict = abstract_attribute()
    _data: np.ndarray = abstract_attribute()

    def __len__(self):
        return len(self._label_dict)

    def _unlabeled_indices(self) -> np.ndarray:
        labeled_keys = np.array(list(self._label_dict.keys()))
        if len(labeled_keys) == 0:
            return np.arange(len(self._data))
        indices = np.delete(np.arange(len(self._data)), labeled_keys)
        return indices

    def D(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._data, np.arange(len(self._data))

    @abstractmethod
    def digest(self, instance_index, new_label):
        raise NotImplementedError

    def label_for_instance(self, index):
        raise NotImplementedError

    def noisy_label_for_instance(self, index):
        raise NotImplementedError

    def has_label_for_instance(self, index: int):
        raise NotImplementedError

    def get_instance(self, index: int):
        raise NotImplementedError

    def U(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def L(self, exclude_index: int = -1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def n_distinct_labels(self) -> int:
        raise NotImplementedError


class Oracle(UsesCustomRNG):
    @abstractmethod
    def get_noisy_label(self, instance: np.ndarray, index: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_clean_label(self, index: int) -> int:
        raise NotImplementedError


class LabelCorrector(UsesCustomRNG):
    name: str = abstract_attribute()

    @abstractmethod
    def _predict(self, instance) -> int:
        raise NotImplementedError

    @abstractmethod
    def is_label_correct(self, instance, label) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_confused(self, instance, label=None):
        raise NotImplementedError

    @abstractmethod
    def _fit(self, data, labels):
        raise NotImplementedError

    @abstractmethod
    def should_relabel(self,
                       data: np.ndarray,
                       labels: np.ndarray,
                       query_instance: np.ndarray,
                       query_label: int) -> bool:
        raise NotImplementedError


class Strategy(UsesCustomRNG):
    name: str = abstract_attribute()
    dataset: Dataset = abstract_attribute()
    query_strategy: QueryStrategy = abstract_attribute()
    oracle: Oracle = abstract_attribute()
    learner = abstract_attribute()

    @abstractmethod
    def execute_round(self):
        raise NotImplementedError

    @abstractmethod
    def num_queries(self) -> int:
        raise NotImplementedError
