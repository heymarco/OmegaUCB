import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from components.asymptotic_regression import M
from components.optimizer import Optimizer
from components.data import Oracle, SymmetricCrowdsourcingOracle, LabeledPool, MajorityVotedLabeledPool
from components.risk_estimation import RiskEstimator, EnsembleRiskEstimator

MNIST = 554
MUSHROOM = 24

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

    seed = 0
    n_0 = 100
    p = 0.4
    oracle = SymmetricCrowdsourcingOracle(y=clean_labels, p=p, seed=seed)
    pool = MajorityVotedLabeledPool()
    meta_model = M(m_0=1 / len(np.unique(clean_labels)),
                   fit_m_0=False)
    risk_estimator = EnsembleRiskEstimator()
    predictor = DecisionTreeClassifier(random_state=seed)
    # predictor = RandomForestClassifier(n_estimators=10)
    optimizer = Optimizer(predictor=predictor,
                          risk_estimator=risk_estimator,
                          oracle=oracle,
                          labeled_pool=pool,
                          data=clean_data,
                          meta_model=meta_model,
                          B=10000,
                          n_0=n_0,
                          t_0=1,
                          seed=seed)

    while optimizer.B > optimizer.n * optimizer.t:
        optimizer.query_labels()
        optimizer.iterate()
    if optimizer.B > 0:
        optimizer.n = int(optimizer.B / optimizer.t)
        if optimizer.n > 0:
            optimizer.query_labels()
            optimizer.iterate()
