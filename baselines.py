import os

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from components.data import SymmetricCrowdsourcingOracle, Oracle
from components.exp_logging import logger


class Baseline:
    def __init__(self,
                 data: np.ndarray,
                 clean_labels: np.ndarray,
                 model,
                 oracle: Oracle,
                 p: float,
                 n: int,
                 B: int,
                 seed: int,
                 name: str = "Topline"):
        self.n = n
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

    def iterate(self):
        for _ in range(self.n):
            y = self.oracle.query(1)
            x = self.data[self.oracle.queried_index()]
            self.x.append(x)
            self.y.append(y)

        self.model.fit(self.x, self.y)
        mask = np.ones(len(self.data), dtype=bool)
        mask[self.oracle.queried_indices] = 0
        x = self.data[mask]
        y = self.clean_labels[mask]
        score = self.model.score(x, y)
        logger.track_true_score(score)
        self.remaining_budget -= self.n
        return score

    def run(self):
        logger.track_approach(self.name)
        while self.remaining_budget > 0:
            performance = self.iterate()
            logger.track_time()
            logger.track_t_n(t=1, n=self.n)
            logger.finalize_round()
            print(self.remaining_budget)
            if performance >= 0.99:
                break


MNIST = 554
MUSHROOM = 24

if __name__ == '__main__':
    ds = MNIST
    clean_data, clean_labels = fetch_openml(data_id=ds, return_X_y=True, as_frame=False)
    not_na_rows = np.any(np.isnan(clean_data), axis=1) == False
    clean_data = clean_data[not_na_rows]
    clean_labels = clean_labels[not_na_rows]

    B = 1000
    logger.track_dataset_name("mushroom" if ds == MUSHROOM else "mnist")
    for p in [0.0, 0.25, 0.5, 0.7]:
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
            predictor = DecisionTreeClassifier(max_depth=5, random_state=seed)
            alg = Baseline(n=10,
                           data=clean_data,
                           clean_labels=clean_labels,
                           model=predictor,
                           oracle=oracle,
                           p=p,
                           B=B,
                           seed=seed)
            alg.run()
    df = logger.get_dataframe()
    df.to_csv(os.path.join(os.getcwd(), "results", "results_baseline.csv"), index=False)
