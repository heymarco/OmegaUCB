from typing import Tuple

import numpy as np

from .abstract import Dataset


class StandardDataset(Dataset):
    def __init__(self, data: np.ndarray, seed):
        self.rng = np.random.default_rng(seed)
        self._label_dict = {}
        self._data = data
        self._L: np.ndarray = None

    def n_data(self):
        return len(self._label_dict)

    def U(self) -> Tuple[np.ndarray, np.ndarray]:
        indices = self._unlabeled_indices()
        return self._data[indices], indices

    def L(self, exclude_index: int = -1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        labeled_keys = np.array(list(self._label_dict.keys()))
        labeled_keys = labeled_keys[labeled_keys != exclude_index]
        labeled_instances: np.ndarray = self._data[labeled_keys]
        labels = self._labels(exclude_index)
        return labeled_instances, labels, labeled_keys

    def _labels(self, exclude_index: int = -1):
        keys = np.array(list(self._label_dict.keys()))
        indices = keys[keys != exclude_index]
        return np.array([self.label_for_instance(i) for i in indices])

    def has_label_for_instance(self, index: int):
        return index in self._label_dict.keys()

    def digest(self, instance_index: int, new_label: int):
        self._label_dict[instance_index] = new_label

    def label_for_instance(self, index: int):
        return self._label_dict[index]

    def noisy_label_for_instance(self, index: int):
        return self.label_for_instance(index)

    def get_instance(self, index: int):
        return self._label_dict[index]

    def n_distinct_labels(self) -> int:
        return len(self._label_dict)


class MajorityVotedDataset(Dataset):
    def __init__(self, data: np.ndarray, seed: int):
        self.rng = np.random.default_rng(seed)
        self._label_dict = {}
        self._data = data
        self._L: np.ndarray = None

    def n_data(self):
        return len(self._label_dict)

    def U(self) -> Tuple[np.ndarray, np.ndarray]:
        indices = self._unlabeled_indices()
        return self._data[indices], indices

    def L(self, exclude_index: int = -1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        labeled_keys = np.array(list(self._label_dict.keys()))
        if len(labeled_keys) == 0:
            return np.array([]), np.array([]), np.array([])
        labeled_keys = labeled_keys[labeled_keys != exclude_index]
        labeled_instances: np.ndarray = self._data[labeled_keys]
        labels = self._labels(exclude_index)
        return labeled_instances, labels, labeled_keys

    def _labels(self, exclude_index: int = -1):
        keys = np.array(list(self._label_dict.keys()))
        indices = keys[keys != exclude_index]
        return np.array([self.label_for_instance(i) for i in indices])

    def has_label_for_instance(self, index: int):
        return index in self._label_dict.keys()

    def digest(self, instance_index: int, new_label: int):
        if not instance_index in self._label_dict.keys():
            self._label_dict[instance_index] = [new_label]
        else:
            self._label_dict[instance_index].append(new_label)

    def label_for_instance(self, index: int):
        all_estimates = self._label_dict[index]
        predictions, counts = np.unique(all_estimates, return_counts=True)
        candidate_indices = [i for i in range(len(counts)) if counts[i] == max(counts)]
        label = predictions[candidate_indices[0]] if len(candidate_indices) == 1 else np.nan
        if np.isnan(label):
            return np.nan
        else:
            return label

    def noisy_label_for_instance(self, index: int):
        return self._label_dict[index][0]

    def get_instance(self, index: int):
        return self._label_dict[index]

    def n_distinct_labels(self) -> int:
        return len(self._label_dict)
