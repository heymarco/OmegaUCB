import os

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from components.data import SymmetricCrowdsourcingOracle, Oracle
from components.exp_logging import ExperimentLogger
from util import linear_cost_function


class Baseline:
    def __init__(self,
                 data: np.ndarray,
                 clean_labels: np.ndarray,
                 model,
                 oracle: Oracle,
                 p: float,
                 n: int,
                 t: int,
                 B: int,
                 seed: int,
                 logger: ExperimentLogger,
                 name: str = "Topline"):
        self.logger = logger
        self.n = n
        self.t = t
        self.B = B
        self.p = p
        self.oracle = oracle
        self.remaining_budget = B
        self.data = data
        self.clean_labels = clean_labels
        self.model = model
        self.rng = np.random.default_rng(seed)
        self.training_mask = np.zeros(shape=len(data), dtype=bool)
        self.name = name
        self.x = []
        self.y = []
        self._score = []

    def stop_early(self, t):
        if len(self._score) < t:
            return False
        return self._score[-1] - self._score[-t] < 0.01

    def cost(self, t):
        return linear_cost_function(t)

    def iterate(self):
        for _ in range(self.n):
            y = self.oracle.query(self.t)
            x = self.data[self.oracle.queried_index()]
            self.x.append(x)
            self.y.append(y)

        self.model.fit(self.x, self.y)
        mask = np.ones(len(self.data), dtype=bool)
        mask[self.oracle.queried_indices] = 0
        x = self.data[mask]
        y = self.clean_labels[mask]
        score = self.model.score(x[:1000], y[:1000])
        self.logger.track_true_score(score)
        self.remaining_budget -= self.n * self.cost(self.t)
        return score

    def run(self):
        self.logger.track_approach(self.name)
        while self.remaining_budget > 0:
            performance = self.iterate()
            # self._score.append(performance)
            # if self.stop_early(3):
            #     break
            self.logger.track_time()
            self.logger.track_t_n(t=self.t, n=self.n)
            self.logger.finalize_round()
            print(self.remaining_budget)
            if performance >= 0.99:
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

    B = 10000
    dfs = []
    ts = [1, 3, 10, 30, 100]
    for t in ts:
        for p in [0.5]:
            for rep in range(3):
                # LOGGING
                logger = ExperimentLogger()
                logger.track_noise_level(p)
                logger.track_dataset_name("mushroom" if ds == MUSHROOM else "mnist")
                logger.track_rep(rep)
                # SETUP
                seed = rep
                rng = np.random.default_rng(seed)
                clean_labels = LabelEncoder().fit_transform(clean_labels)
                shuffled_indices = np.arange(len(clean_labels))
                rng.shuffle(shuffled_indices)
                clean_data = clean_data[shuffled_indices]
                clean_labels = clean_labels[shuffled_indices]
                oracle = SymmetricCrowdsourcingOracle(y=clean_labels, p=p, seed=seed)
                predictor = DecisionTreeClassifier()
                # ALGORITHM
                alg = Baseline(n=100,
                               t=t,
                               data=clean_data,
                               clean_labels=clean_labels,
                               model=predictor,
                               oracle=oracle,
                               p=p,
                               B=B,
                               logger=logger,
                               name="Baseline-{}".format(t),
                               seed=seed)
                df = alg.run()
                # APPEND RESULT
                dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(os.getcwd(), "results", "results_baseline.csv"), index=False)
