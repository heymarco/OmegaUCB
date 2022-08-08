import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .abstract import LabelCorrector
from experiment_logging import logger


class DummyLC(LabelCorrector):
    def __init__(self):
        self.name = "dummy"
        self.rng = None

    def _predict(self, instance) -> int:
        pass

    def is_label_correct(self, instance, label) -> bool:
        return True

    def is_confused(self, instance, label=None):
        return False, np.nan

    def _fit(self, data, labels):
        pass

    def should_relabel(self,
                       data: np.ndarray,
                       labels: np.ndarray,
                       query_instance: np.ndarray,
                       query_label: int) -> bool:
        return False


class RandomForestLC(LabelCorrector):
    def __init__(self, confusion_thresh: float, seed: int):
        self.name = "agreement"
        self.rng = np.random.default_rng(seed)
        self.confusion_threshold = confusion_thresh
        self.model = RandomForestClassifier(max_depth=2)

    def _fit(self, data, labels):
        self.model.fit(data, labels)

    def _predict(self, instance) -> int:
        return self.model.predict([instance])[0]

    def is_label_correct(self, instance, label) -> bool:
        return self.model.predict([instance])[0] == label

    def is_confused(self, instance, label=None):
        proba = self.model.predict_proba([instance])[0]
        confusion_score = 1 - max(proba)
        return confusion_score > self.confusion_threshold, confusion_score

    def should_relabel(self, data: np.ndarray,
                       labels: np.ndarray,
                       query_instance: np.ndarray,
                       query_label: int) -> bool:
        self._fit(data, labels)
        label_correct = self.is_label_correct(query_instance, query_label)
        confused, score = self.is_confused(query_instance)
        print(score, label_correct)
        return not label_correct and confused


class BinaryLC(LabelCorrector):
    def __init__(self, num_classes: int, proba_thresh: float, seed: int):
        self.num_classes = num_classes
        self.name = "binary"
        self.proba_thresh = proba_thresh
        self.rng = np.random.default_rng(seed)
        self.model = RandomForestClassifier()

    def _fit(self, data, labels):
        self.model.fit(data, labels)

    def _predict(self, instance) -> int:
        return self.model.predict([instance])[0]

    def is_label_correct(self, instance, label) -> bool:
        return self.model.predict([instance])[0] == label

    def is_confused(self, instance, label=None):
        instance = np.append(instance, [label])
        proba = self.model.predict_proba([instance])[0]
        return proba < self.proba_thresh, proba

    def should_relabel(self, data: np.ndarray,
                       labels: np.ndarray,
                       query_instance: np.ndarray,
                       query_label: int) -> bool:
        ext_data = []
        bin_labels = []
        for instance, label in zip(data, labels):
            extended_instance = np.append(instance, [label], axis=0)
            ext_data.append(extended_instance)
            bin_labels.append(True)
            all_labels = np.arange(self.num_classes)
            false_label = np.random.choice(all_labels[all_labels != label])
            extended_instance = np.append(instance, [false_label], axis=0)
            ext_data.append(extended_instance)
            bin_labels.append(False)
        self._fit(ext_data, bin_labels)
        extended_query_instance = np.append(query_instance, [query_label], axis=0)
        is_correct = self._predict(extended_query_instance)
        proba = np.max(self.model.predict_proba([extended_query_instance]))
        if proba > self.proba_thresh and not is_correct:
            print("relabel", proba)
        else:
            print(proba)
        return not is_correct and proba > self.proba_thresh


