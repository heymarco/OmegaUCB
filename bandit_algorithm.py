import os

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from components.data import Oracle, SymmetricCrowdsourcingOracle
from components.exp_logging import ExperimentLogger
from components.path import Path, PathElement
from components.risk_estimation import RiskEstimator, EnsembleRiskEstimator


class Bandit:
    def __init__(self):
        self.alpha = 0.0
        self.beta = 0.0

    def sample(self):
        return np.random.beta(a=self.alpha + 1, b=self.beta + 1)

    def update(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta


class BanditLabeler:
    def __init__(self,
                 t_max: int,
                 n: int,
                 oracle: Oracle,
                 risk_estimator: RiskEstimator,
                 data: np.ndarray,
                 clean_labels: np.ndarray,
                 model, budget: float,
                 name: str,
                 logger: ExperimentLogger,
                 use_validation_labels: bool = False):
        self.use_validation_labels = use_validation_labels
        self.name = name
        self.logger = logger
        self.t_max = t_max
        self.n = n
        self.oracle = oracle
        self.risk_estimator = risk_estimator
        self.data = data
        self.clean_labels = clean_labels
        self.model = model
        self.budget = budget
        self.remaining_budget = budget
        self.bandits = [Bandit() for _ in range(t_max)]
        self.path = Path()

    def iterate(self):
        self._labeling()
        if len(self.path) > 1:
            l, ts = self._estimate_reward(use_labels=self.use_validation_labels)
            self._update_parameters(l, ts)

    def _labeling(self):
        theta = [b.sample() for b in self.bandits]
        t = int(np.argmax(theta)) + 1
        x = []
        y = []
        for _ in range(self.n):
            y.append(self.oracle.query(t))
            y_index = self.oracle.queried_index()
            x.append(self.data[y_index])
        x = np.vstack(x)
        y = np.vstack(y)
        element = PathElement(data=(x, y), time=t)
        self.path.elements.append(element)
        self.remaining_budget -= self.n * t
        self.logger.track_t_n(t, self.n)

    def _unlabeled_data(self):
        mask = np.ones(shape=len(self.data), dtype=bool)
        mask[self.oracle.queried_indices] = False
        return self.data[mask]

    def _evaluate_path(self, path, use_true_labels: bool = True):
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

    def _estimate_reward(self, use_labels: bool = True):
        l = []
        ts = []
        l_now = self._evaluate_path(self.path, use_true_labels=use_labels)
        for i in range(len(self.path)):
            t = self.path.elements[i].time
            ts.append(t)
            subpath = self.path.subpath_without_index(i)
            l_t = self._evaluate_path(subpath, use_true_labels=use_labels)
            rho = l_now - l_t
            reg = 0.2
            rho = rho * (1 - reg * (t - 1) / self.t_max)
            l.append(rho)
        return l, ts

    def _update_parameters(self, l, ts):
        ts = np.array(ts)
        l = np.array(l)
        l = l - np.min(l)
        if np.max(l) > 0:
            l = l / np.max(l)
        for t in np.unique(ts):
            mask = ts == t
            n_paths_t = len(ts[mask])
            reward_sum_t = float(np.sum(l[mask]))
            if reward_sum_t > 0:
                a = reward_sum_t
                b = n_paths_t - reward_sum_t
                self.bandits[t-1].update(a, b)

    def run(self):
        self.logger.track_approach(self.name)
        while self.remaining_budget > 0:
            self.iterate()
            predicted_performance = self._evaluate_path(self.path, use_true_labels=False)
            true_performance = self._evaluate_path(self.path, use_true_labels=True)
            self.logger.track_true_score(true_performance)
            self.logger.track_estimated_score(predicted_performance)
            self.logger.track_alpha_beta(alpha=np.array([b.alpha for b in self.bandits]),
                                    beta=np.array([b.beta for b in self.bandits]))
            self.logger.track_time()
            print("predicted performance: {}, true performance: {}".format(predicted_performance, true_performance))
            print("remaining budget: {}".format(self.remaining_budget))
            self.logger.finalize_round()
            if true_performance >= 0.99:
                break
        return self.logger.get_dataframe()


MNIST = 554
MUSHROOM = 24

if __name__ == '__main__':
    ds = MNIST
    clean_data, clean_labels = fetch_openml(data_id=ds, return_X_y=True, as_frame=False)
    not_na_rows = np.any(np.isnan(clean_data), axis=1) == False
    clean_data = clean_data[not_na_rows]
    clean_labels = clean_labels[not_na_rows]

    B = 1000
    dfs = []
    for use_val_labels in [True, False]:
        for p in [0.0, 0.25, 0.5, 0.7]:
            for rep in range(1):
                logger = ExperimentLogger()
                logger.track_dataset_name("mushroom" if ds == MUSHROOM else "mnist")
                logger.track_noise_level(p)
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
                predictor = DecisionTreeClassifier(max_depth=5, random_state=seed)
                alg = BanditLabeler(n=10,
                                    t_max=5,
                                    oracle=oracle,
                                    risk_estimator=risk_estimator,
                                    data=clean_data,
                                    clean_labels=clean_labels,
                                    model=predictor,
                                    budget=B,
                                    name="Bandit-{}".format("L" if use_val_labels else "U"),
                                    use_validation_labels=use_val_labels,
                                    logger=logger)
                df = alg.run()
                dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(os.getcwd(), "results", "results_bandit.csv"), index=False)
