from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


class RiskEstimator(ABC):

    @abstractmethod
    def prepare(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def estimate_risk(self, *args, **kwargs):
        raise NotImplementedError


class EnsembleRiskEstimator(RiskEstimator):
    def __init__(self, ensemble_type="rf", n_estimators=100):
        if ensemble_type == "rf":
            self.ensemble = RandomForestClassifier(n_estimators=n_estimators)
        elif ensemble_type == "dt":
            self.ensemble = DecisionTreeClassifier()

    def prepare(self, *args, **kwargs):
        x_train = kwargs["x_train"]
        y_train = kwargs["y_train"]
        self.ensemble.fit(x_train, y_train)

    def estimate_risk(self, *args, **kwargs):
        x_test = kwargs["x_test"]
        model = kwargs["model"]
        y_model = model.predict(x_test)
        y_ensemble = self.ensemble.predict(x_test)
        agreement = np.sum(y_model == y_ensemble)
        acc_estimate = agreement / len(x_test)
        return acc_estimate


MUSHROOM = 24
MNIST = 554


if __name__ == '__main__':
    split = 0.01
    clean_data, clean_labels = fetch_openml(data_id=MUSHROOM, return_X_y=True, as_frame=False)
    not_na_rows = np.any(np.isnan(clean_data), axis=1) == False
    clean_data = clean_data[not_na_rows]
    clean_labels = clean_labels[not_na_rows]
    shuffled_indices = np.arange(len(clean_labels))
    np.random.shuffle(shuffled_indices)
    clean_data = clean_data[shuffled_indices]
    clean_labels = clean_labels[shuffled_indices]
    clean_labels = LabelEncoder().fit_transform(clean_labels)
    n_train = int(len(clean_data) * split)
    x_train, x_test = clean_data[:n_train], clean_data[n_train:]
    y_train, y_test = clean_labels[:n_train], clean_labels[n_train:]
    risk_estimator = EnsembleRiskEstimator()

    model = DecisionTreeClassifier(max_depth=5)
    model.fit(x_train, y_train)
    risk_estimator.prepare(x_train=x_train, y_train=y_train)

    theta_mle = []
    acc_scores = []
    empirical_error = []
    sample_range = range(10, 2000, 20)
    for sample_size in tqdm(sample_range):
        sample_indices = np.random.choice(np.arange(len(x_test)), sample_size)
        acc = risk_estimator.estimate_risk(model=model, x_test=x_test[sample_indices])
        predictions = model.predict(x_test[sample_indices])
        labels = y_test[sample_indices]
        acc_scores.append(model.score(x_test[sample_indices], labels))
        empirical_error.append(acc)
    acc_scores = np.array(acc_scores)
    empirical_error = np.array(empirical_error)

    plt.plot(sample_range, empirical_error, color="orange", label="empirical error")
    plt.plot(sample_range, acc_scores, color="blue", label="true error")
    plt.legend()
    plt.tight_layout(pad=.5)
    plt.show()
