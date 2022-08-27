import numpy as np

from components.bandit import ThompsonSampling
from components.data import Oracle
from components.exp_logging import ExperimentLogger
from components.path import Path, PathElement
from components.risk_estimation import RiskEstimator
from util import linear_cost_function


class BanditLabeling:
    def __init__(self,
                 times,
                 n: int,
                 oracle: Oracle,
                 risk_estimator: RiskEstimator,
                 data: np.ndarray,
                 clean_labels: np.ndarray,
                 model,
                 budget: float,
                 name: str,
                 logger: ExperimentLogger,
                 seed: int,
                 use_validation_labels: bool = False):
        self.use_validation_labels = use_validation_labels
        self.name = name
        self.logger = logger
        self.times = np.sort(times)
        self.n = n
        self.oracle = oracle
        self.risk_estimator = risk_estimator
        self.data = data
        self.clean_labels = clean_labels
        self.model = model
        self.budget = budget
        self.remaining_budget = budget
        self.bandit = ThompsonSampling(k=len(times))
        self.path = Path(elements=[])
        self.rng = np.random.default_rng(seed)
        self._score = []
        self.r = 0

    def cost(self, t):
        return linear_cost_function(t)

    def iterate(self):
        self._labeling()
        if len(self.path) > 1:
            l, cs, arms = self._estimate_reward(use_labels=self.use_validation_labels)
            self._update_parameters(l, cs, arms)
        self.r += 1

    def _labeling(self):
        if self.r < len(self.bandit):  # pull each arm once
            t = self.times[self.r]
            index = self.r
        else:
            index = self.bandit.sample()
            t = self.times[index]
        x = []
        y = []
        for _ in range(self.n):
            y.append(self.oracle.query(t))
            y_index = self.oracle.queried_index()
            x.append(self.data[y_index])
        x = np.vstack(x)
        y = np.vstack(y)
        element = PathElement(data=(x, y), time=t, index=index)
        self.path.elements.append(element)
        self.remaining_budget -= self.n * self.cost(t)
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
            score = self.model.score(x[:1000], y[:1000])
            return score
        x_full, y_full = self.path.data()
        x, y = path.data()
        self.model.fit(x, y)
        self.risk_estimator.fit(x_full, y_full)
        return self.risk_estimator.predict(self._unlabeled_data()[:1000], self.model)

    def _estimate_reward(self, use_labels: bool = True):
        l = []
        ts = []
        arms = []
        l_now = self._evaluate_path(self.path, use_true_labels=use_labels)
        for i in range(len(self.path)):
            t = self.path.elements[i].time
            a = self.path.elements[i].arm_index
            subpath = self.path.subpath_without_index(i)
            l_t = self._evaluate_path(subpath, use_true_labels=use_labels)
            rho = l_now - l_t  # normalize to 0, 1
            l.append(rho)
            arms.append(a)
            ts.append(t)
        return l, ts, arms

    def _update_parameters(self, l, ts, arms):
        ts = np.array(ts)
        l = np.array(l)
        reward_sums = {}
        total_costs = {}
        for i in arms:
            reward_sums[i] = 0
            total_costs[i] = 0
        # Sum up rewards for tested arms
        for i, arm in enumerate(arms):
            total_costs[arm] += self.cost(ts[i])
            reward_sums[arm] += float(l[i])  # this is in line with Budgeted Thompson Sampling
        # Cancel out negative rewards by setting them to 0
        avg_rewards = {}
        for arm in reward_sums:
            reward_sums[arm] = np.maximum(0, reward_sums[arm])
            # Compute average rewards
            avg_rewards[arm] = reward_sums[arm] / total_costs[arm]
        # Update parameters if the reward is positive for any arm (edge case)
        if np.max(list(avg_rewards.values())) <= 0:
            return
        # Scale rewards to [0,1]
        max_reward = np.max(list(avg_rewards.values()))
        for arm in avg_rewards:
            avg_rewards[arm] = avg_rewards[arm] / max_reward
        # Multiply by number the respective arms had
        unique_arms, pulls = np.unique(arms, return_counts=True)
        for arm in unique_arms:
            avg_rewards[arm] = avg_rewards[arm] * pulls[arm]
        rewards = avg_rewards
        print(rewards)
        # Update bandit
        for i, arm in enumerate(unique_arms):
            alpha = rewards[arm]
            beta = pulls[arm] - rewards[arm]
            self.bandit.set(i, alpha, beta)

    def run(self):
        self.logger.track_approach(self.name)
        while self.remaining_budget > 0:
            self.iterate()
            predicted_performance = self._evaluate_path(self.path, use_true_labels=False)
            true_performance = self._evaluate_path(self.path, use_true_labels=True)
            self.logger.track_true_score(true_performance)
            self.logger.track_estimated_score(predicted_performance)
            self.logger.track_alpha_beta(alpha=self.bandit.alphas(), beta=self.bandit.betas())
            self.logger.track_time()
            print("predicted performance: {}, true performance: {}".format(predicted_performance, true_performance))
            print("remaining budget: {}".format(self.remaining_budget))
            self.logger.finalize_round()
        return self.logger.get_dataframe()
