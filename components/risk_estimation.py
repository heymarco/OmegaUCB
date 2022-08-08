from abc import ABC, abstractmethod

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class RiskEstimator(ABC):
    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError


class EnsembleRiskEstimator(RiskEstimator):
    def __init__(self, ensemble_type="rf", n_estimators=100):
        if ensemble_type == "rf":
            self.ensemble = RandomForestClassifier(n_estimators=n_estimators)
        elif ensemble_type == "dt":
            self.ensemble = DecisionTreeClassifier()

    def fit(self, *args, **kwargs):
        x_train = kwargs["x_train"]
        y_train = kwargs["y_train"]
        self.ensemble.fit(x_train, y_train)

    def predict(self, *args, **kwargs):
        x_test = kwargs["x_test"]
        model = kwargs["model"]
        y_model = model.predict(x_test)
        y_ensemble = self.ensemble.predict(x_test)
        agreement = np.sum(y_model == y_ensemble)
        acc_estimate = agreement / len(x_test)
        return acc_estimate
