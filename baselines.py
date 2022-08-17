import os

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from exp_logging import logger


class Topline:
    def __init__(self,
                 data: np.ndarray,
                 clean_labels: np.ndarray,
                 model,
                 n: int,
                 B: int,
                 seed: int,
                 name: str = "Topline"):
        self.n = n
        self.B = B
        self.remaining_budget = B
        self.data = data
        self.clean_labels = clean_labels
        self.model = model
        self.rng = np.random.default_rng(seed)
        self.training_mask = np.zeros(shape=len(data), dtype=bool)
        self.name = name

    def iterate(self):
        unlabeled = np.arange(len(self.data))[np.invert(self.training_mask)]
        labeled_indices = self.rng.choice(unlabeled, self.n)
        self.training_mask[labeled_indices] = True
        x = self.data[self.training_mask]
        y = self.clean_labels[self.training_mask]
        self.model.fit(x, y)
        score = self.model.score(self.data[np.invert(self.training_mask)],
                                 self.clean_labels[np.invert(self.training_mask)])
        logger.track_true_score(score)
        self.remaining_budget -= self.n
        return score

    def run(self):
        logger.track_approach(self.name)
        while self.remaining_budget > 0:
            performance = self.iterate()
            logger.track_time()
            logger.finalize_round()
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
    for p in [0.0]:
        logger.track_noise_level(p)
        for rep in range(1):
            seed = rep
            rng = np.random.default_rng(seed)
            clean_labels = LabelEncoder().fit_transform(clean_labels)
            shuffled_indices = np.arange(len(clean_labels))
            rng.shuffle(shuffled_indices)
            clean_data = clean_data[shuffled_indices]
            clean_labels = clean_labels[shuffled_indices]
            logger.track_rep(rep)
            predictor = DecisionTreeClassifier(max_depth=5, random_state=seed)
            alg = Topline(n=10,
                          data=clean_data,
                          clean_labels=clean_labels,
                          model=predictor,
                          B=B,
                          seed=seed)
            alg.run()
    df = logger.get_dataframe()
    df.to_csv(os.path.join(os.getcwd(), "..", "results", "results_topline.csv"), index=False)
