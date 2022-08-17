import os
from typing import Tuple

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from components.data import Oracle, SymmetricCrowdsourcingOracle
from components.path import PathElement, Path
from components.risk_estimation import RiskEstimator, EnsembleRiskEstimator
from components.exp_logging import logger


class IterativePathGradientEstimator:
    def __init__(self,
                 oracle: Oracle,
                 risk_estimator: RiskEstimator,
                 data: np.ndarray,
                 clean_labels: np.ndarray,
                 model, budget: float,
                 n_min: float = 10):
        self.oracle = oracle
        self.model = model
        self.budget = budget
        self.data = data
        self.clean_labels = clean_labels
        self.path = Path()
        self.n_min = n_min
        self.risk_estimator = risk_estimator
        self.remaining_budget = budget

        for t in [1, 2, 5, 10]:
            self.label_batch(t, self.n_min)

    def label_batch(self, t: float, n: int):
        x = []
        y = []
        for _ in range(n):
            y.append(self.oracle.query(t))
            y_index = self.oracle.queried_index()
            x.append(self.data[y_index])
        x = np.vstack(x)
        y = np.vstack(y)
        element = PathElement(data=(x, y), time=t)
        self.path.elements.append(element)
        self.remaining_budget -= n * t

    def _unlabeled_data(self):
        mask = np.ones(shape=len(self.data), dtype=bool)
        mask[self.oracle.queried_indices] = False
        return self.data[mask]

    def evaluate_path(self, path, use_true_labels: bool = True):
        if use_true_labels:
            x, y = path.data()
            self.model.fit(x, y)
            mask = np.ones(len(self.data), dtype=bool)
            mask[self.oracle.queried_indices] = 0
            x = self.data[mask]
            y = self.clean_labels[mask]
            score = self.model.score(x, y)
            return score
        x_full, y_full = self.path.data()
        x, y = path.data()
        self.risk_estimator.fit(x_full, y_full)
        self.model.fit(x, y)
        return self.risk_estimator.predict(self._unlabeled_data(), self.model)

    def compute_gradients_vectors(self):
        current_performance = self.evaluate_path(self.path)
        perf_shorter_paths = []
        vectors = []
        for i in range(len(self.path)):
            subpath = self.path.subpath_without_index(i)
            v = subpath.vector()
            v = self.path.vector() - v
            perf = self.evaluate_path(subpath)
            perf_shorter_paths.append(perf)
            vectors.append(v)
        vectors = np.array(vectors)
        perf_diff = current_performance - np.array(perf_shorter_paths)
        gradients = perf_diff / (vectors[:, 0] * vectors[:, 1])
        return gradients, vectors

    def adapt_direction(self, direction) -> np.ndarray:
        if direction[1] == 1:
            direction[0] += 1
        return direction

    def get_new_direction(self, strategy="avg") -> np.ndarray:
        gradients, vectors = self.compute_gradients_vectors()
        logger.track_gradients(gradients)
        logger.track_vectors(vectors)
        gradients = np.maximum(0, gradients)  # we only look at positive gradients
        gradients = gradients / np.max(gradients)
        gradients = gradients / np.sum(gradients)
        vectors = np.array(vectors)
        for i in range(len(vectors)):
            vectors[i] *= gradients[i]
        if strategy == "max":
            new_direction = vectors[np.argmax(gradients)]
        else:
            new_direction = np.sum(vectors, axis=0)
            new_direction = self.adapt_direction(new_direction)
        new_direction = new_direction / new_direction[1]
        return new_direction

    def get_t_n(self):
        direction = self.get_new_direction()
        if np.isnan(direction[0]):
            direction[0] = 1 / self.n_min
        if np.isnan(direction[1]):
            direction[1] = 1
        direction *= self.n_min
        print(direction)
        return direction

    def evaluate(self) -> float:
        mask = np.ones(len(self.data), dtype=bool)
        mask[self.oracle.queried_indices] = 0
        x = self.data[mask]
        y = self.clean_labels[mask]
        score = self.model.score(x, y)
        return score


    def run(self):
        while self.remaining_budget > 0:
            t, n = self.get_t_n()
            t = round(t)
            n = int(n)
            self.label_batch(t, n)
            logger.track_t_n(t, n)
            predicted_performance = self.evaluate_path(self.path)
            true_performance = self.evaluate()
            logger.track_true_score(true_performance)
            logger.track_estimated_score(predicted_performance)
            logger.track_time()
            print("predicted performance: {}, true performance: {}".format(predicted_performance, true_performance))
            print("remaining budget: {}".format(self.remaining_budget))
            logger.finalize_round()
            if true_performance >= 0.99:
                break

MNIST = 554
MUSHROOM = 24

if __name__ == '__main__':
    ds = MNIST
    clean_data, clean_labels = fetch_openml(data_id=ds, return_X_y=True, as_frame=False)
    not_na_rows = np.any(np.isnan(clean_data), axis=1) == False
    clean_data = clean_data[not_na_rows]
    clean_labels = clean_labels[not_na_rows]

    B = 500
    logger.track_dataset_name("mushroom" if ds == MUSHROOM else "mnist")
    for p in [0.0, 0.2, 0.4]:
        logger.track_noise_level(p)
        for rep in range(3):
            seed = rep
            rng = np.random.default_rng(seed)
            clean_labels = LabelEncoder().fit_transform(clean_labels)
            shuffled_indices = np.arange(len(clean_labels))
            rng.shuffle(shuffled_indices)
            clean_data = clean_data[shuffled_indices]
            clean_labels = clean_labels[shuffled_indices]
            logger.track_rep(rep)
            oracle = SymmetricCrowdsourcingOracle(y=clean_labels, p=p, seed=seed)
            risk_estimator = EnsembleRiskEstimator()
            predictor = DecisionTreeClassifier(random_state=seed)
            alg = IterativePathGradientEstimator(oracle=oracle,
                                                 risk_estimator=risk_estimator,
                                                 data=clean_data,
                                                 clean_labels=clean_labels,
                                                 model=predictor,
                                                 budget=B)
            alg.run()
    df = logger.get_dataframe()
    df.to_csv(os.path.join(os.getcwd(), "results", "results.csv"), index=False)
