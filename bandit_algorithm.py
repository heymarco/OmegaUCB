import os

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from components.labeling import BanditLabeling
from components.data import SymmetricCrowdsourcingOracle
from components.exp_logging import ExperimentLogger
from components.risk_estimation import EnsembleRiskEstimator
from util import MNIST, MUSHROOM

if __name__ == '__main__':
    ds = MNIST
    if ds == MUSHROOM or ds == MNIST:
        clean_data, clean_labels = fetch_openml(data_id=ds, return_X_y=True, as_frame=False)
    else:
        clean_data, clean_labels = make_classification(
            n_samples=3000, n_features=5, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
        )
    not_na_rows = np.any(np.isnan(clean_data), axis=1) == False
    clean_data = clean_data[not_na_rows]
    clean_labels = clean_labels[not_na_rows]

    B = 10000
    dfs = []
    for use_val_labels in [True, False]:
        for p in [0.5]:
            for rep in range(3):
                logger = ExperimentLogger()
                if ds == MUSHROOM:
                    logger.track_dataset_name("mushroom")
                elif ds == MNIST:
                    logger.track_dataset_name("mnist")
                else:
                    logger.track_dataset_name("synth")
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
                predictor = DecisionTreeClassifier()
                alg = BanditLabeling(n=100,
                                     times=np.array([1, 3, 10, 30, 100]),
                                     oracle=oracle,
                                     risk_estimator=risk_estimator,
                                     data=clean_data,
                                     clean_labels=clean_labels,
                                     model=predictor,
                                     budget=B,
                                     name="ThompsonSampling-{}".format("L" if use_val_labels else "U"),
                                     use_validation_labels=use_val_labels,
                                     logger=logger,
                                     seed=seed)
                df = alg.run()
                dfs.append(df)
                del alg
                del oracle
                del predictor
                del risk_estimator
                del logger
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(os.getcwd(), "results", "results_bandit.csv"), index=False)
