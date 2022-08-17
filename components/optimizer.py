import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import beta
from sklearn.exceptions import NotFittedError

from components.asymptotic_regression import AsymptoticFunction, M
from components.data import Oracle, LabeledPool
from components.risk_estimation import RiskEstimator


def quality_function(x, K, q_min):
    return q_min + (1 - q_min) * (1 - np.exp(-K * x))


class Optimizer:
    def __init__(self,
                 predictor,
                 risk_estimator: RiskEstimator,
                 meta_model: AsymptoticFunction,
                 oracle: Oracle,
                 labeled_pool: LabeledPool,
                 data: np.ndarray,
                 t_0: float,
                 n_0: int,
                 B: int,
                 seed: int):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.predictor = predictor
        self.risk_estimator = risk_estimator
        self.meta_model = meta_model
        self.oracle = oracle
        self.labeled_pool = labeled_pool
        self.unlabeled_data = data
        self._unlabeled_instances = np.arange(len(self.unlabeled_data))
        self.t_0 = t_0
        self.n_0 = n_0
        self.t = t_0
        self.n = n_0
        self.m_0 = 1 / self.oracle.n_classes
        self._x_prev = None
        self._y_prev = None
        self.B_0 = B
        self.B = B
        self._q_hist = []
        self._t_hist = []

    def query_labels(self):
        self._x_prev = self.labeled_pool.x()
        self._y_prev = self.labeled_pool.y()
        for _ in range(self.n):
            y = self.oracle.query(t=self.t)
            index = self.oracle.queried_index()
            self._unlabeled_instances = self._unlabeled_instances[self._unlabeled_instances != index]
            self.labeled_pool.add(self.t, (self.unlabeled_data[index], y))
        self.B -= self.t * self.n

    def iterate(self):
        mask = np.ones(len(self.unlabeled_data), dtype=bool)
        mask[self.oracle.queried_indices] = False
        test_data = self.unlabeled_data[mask]

        x = self.labeled_pool.x_t(self.t)
        y = self.labeled_pool.y_t(self.t)

        samples = np.arange(2, len(y), 3)
        accs = []
        # train model
        self.risk_estimator.fit(x_train=self.labeled_pool.x(), y_train=self.labeled_pool.y())
        if len(self._y_prev):
            self.predictor.fit(self._x_prev, self._y_prev)
            self.m_0 = self.risk_estimator.predict(x_test=test_data, model=self.predictor)
        for s in samples:
            sample_accs = []
            for _ in range(5):
                sample = self.rng.integers(0, len(y), size=s)
                x_sample = x[sample]
                y_sample = y[sample]
                if len(self._y_prev):
                    x_sample = np.concatenate([self._x_prev, x_sample], axis=0)
                    y_sample = np.concatenate([self._y_prev, y_sample], axis=0)
                self.predictor.fit(x_sample, y_sample)
                acc = self.risk_estimator.predict(x_test=test_data, model=self.predictor)
                sample_accs.append(acc)
            accs.append(np.mean(sample_accs))
        # estimate parameters
        if isinstance(self.meta_model, M):
            self.meta_model.m_0 = self.m_0
            self.meta_model.fit_m_0 = False
            params, _ = curve_fit(
                lambda x_vals, f_max, y_50, power: self.meta_model(x_vals, self.m_0, f_max, y_50, power),
                samples, accs, bounds=[(self.m_0, 0, 0), (1.0, np.infty, np.infty)])  # m_0, f_max, y_50, power
            q_max = params[0]
            model_params = np.concatenate([[self.m_0], params])
            self._t_hist.append(self.t)
            self._q_hist.append(q_max)
            params, _ = curve_fit(
                lambda x, K: quality_function(x, K, self.m_0),
                [self.t], [q_max], bounds=(0, np.infty))
            params = np.concatenate([params, [self.m_0]])
        # optimize
        t_candidates = np.arange(1, 1000)
        remaining_sample_size = self.B / t_candidates
        quality_estimates = quality_function(t_candidates, *params)
        performance_estimates = []
        for t, q, n in zip(t_candidates, quality_estimates, remaining_sample_size):
            model_params[1] = q
            performance = self.meta_model(n, *model_params)
            performance_estimates.append(performance)
        self.t = t_candidates[np.argmax(performance_estimates)]
        print("t_opt is {},".format(self.t),
              "remaining budget is {},".format(self.B),
              "size of L is {}".format(len(self.labeled_pool.y())))
