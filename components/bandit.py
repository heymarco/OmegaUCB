import numpy as np
from scipy import stats

from components.bandits.abstract import AbstractArm, AbstractBandit


class ArmWithAdaptiveBetaPosterior(AbstractArm):
    def __init__(self, seed: int, ci: str):
        self.rng = np.random.default_rng(seed)
        self.alpha = 0.0
        self.beta = 0.0
        self.startup_complete = False
        self.prev_avg = np.nan
        self.this_avg = np.nan
        self.pulls = 0
        self._ci = ci
        self.t = 0

    def __len__(self):
        return self.pulls

    def exp_decay_alpha(self, alpha_max=0.1, k=0.01):
        alpha = alpha_max * (1 - np.exp(-k * (self.t - 1)))
        return max(1e-5, min(1 - 1e-5, alpha))  # avoid singularities

    def sample(self):
        s = self.rng.beta(a=self.alpha + 1, b=self.beta + 1)
        if self._ci == "damped":
            s = 1 / self.pulls * 0.5 + (1 - 1 / self.pulls) * s
        return s

    def mean(self):
        return (self.alpha + 1) / (self.alpha + self.beta + 2)

    def set(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    def update(self, new_val: float, was_pulled: bool):
        self.t += 1
        if was_pulled:
            if not self.startup_complete:
                self.prev_avg = new_val
                self.this_avg = new_val
                self.pulls += 1
                self.startup_complete = True
            else:
                self.prev_avg = self.this_avg
                new_avg = (self.pulls * self.prev_avg + new_val) / (self.pulls + 1)
                self.this_avg = new_avg
                self.pulls += 1
        if self.startup_complete:
            alpha, beta = self._compute_alpha_beta()
            self.set(alpha, beta)

    def hoeffding_ci(self, alpha=0.05):
        return np.sqrt(-1 / (2 * self.pulls) * np.log(alpha / 2))

    def hoeffding_ci_t(self):
        return np.sqrt(np.log(self.t) / self.pulls)

    def baseline_ci(self):
        return 1 / self.pulls

    def jeffrey_ci(self, alpha=0.05):
        n = self.pulls
        x = self.this_avg * n
        low, high = stats.beta.interval(alpha=alpha, a=0.5 + x, b=0.5 + n - x)
        return high

    def get_ci(self):
        if self._ci == "hoeffding":
            return self.hoeffding_ci()
        elif self._ci == "hoeffding-t":
            return self.hoeffding_ci_t()
        elif self._ci == "baseline":
            return self.baseline_ci()
        elif self._ci is None or self._ci == "damped":
            return 0.0
        else:
            raise ValueError

    def compute_wilson_avg(self, alpha):
        ns = self.this_avg * self.pulls
        n = self.pulls
        z = 1.96 if alpha == 0.05 else stats.norm.interval(1 - alpha)[1]
        z2 = z ** 2
        estimate = (ns + 0.5 * z2) / (n + z2)
        if np.isnan(estimate):
            print("")
        return estimate

    def compute_wilson_avg_t(self, alpha_max=0.1):
        alpha = self.exp_decay_alpha(alpha_max=alpha_max)  #  1 / (1 + self.t * np.log(self.t) ** 2)
        return self.compute_wilson_avg(alpha)

    def compute_wilson(self, alpha=0.05):
        ns = self.this_avg * self.pulls
        n = self.pulls
        z = 1.96 if alpha == 0.05 else stats.norm.interval(1 - alpha)[1]
        z2 = z ** 2
        estimate = (ns + 0.5 * z2) / (n + z2)
        ci = 2 / (n + z2) * np.sqrt((ns * (n - ns)) / n + z2 / 4)
        if np.isnan(estimate):
            print("")
        return estimate + ci

    def compute_wilson_t(self, alpha_max=0.1):
        alpha = self.exp_decay_alpha(alpha_max=alpha_max)  # 1 / (1 + self.t * np.log(self.t) ** 2)
        return self.compute_wilson(alpha)

    def _compute_sample_average_estimator(self):
        ci = self.get_ci()
        avg = self.this_avg + ci
        return min(1.0, max(0.0, avg))

    def _compute_alpha_beta(self):
        if self._ci == "wilson-ci":
            avg = self.compute_wilson()
        elif self._ci == "wilson-ci-t":
            avg = self.compute_wilson_t()
        elif self._ci == "wilson":
            avg = self.compute_wilson_avg(alpha=0.05)
        elif self._ci == "wilson-t":
            avg = self.compute_wilson_avg_t()
        elif self._ci == "jeffrey-ci":
            avg = self.jeffrey_ci()
        else:
            avg = self.this_avg + self.get_ci()
        avg = min(1.0, max(0.0, avg))
        alpha = avg * self.pulls
        beta = self.pulls * (1 - avg)
        return alpha, beta


class AdaptiveBudgetedThompsonSampling(AbstractBandit):
    def __init__(self, k: int, name: str, seed: int, ci_reward: str, ci_cost: str):
        self.ci_reward = ci_reward
        self.ci_cost = ci_cost
        self.reward_arms = [ArmWithAdaptiveBetaPosterior(arm_index, ci=ci_reward) for arm_index in range(k)]
        self.cost_arms = [ArmWithAdaptiveBetaPosterior(k + arm_index, ci=ci_cost) for arm_index in range(k)]
        super(AdaptiveBudgetedThompsonSampling, self).__init__(k, name, seed)

    def sample(self) -> int:
        if np.any([not a.startup_complete for a in self.reward_arms]):
            return [i for i, a in enumerate(self.reward_arms) if not a.startup_complete][0]
        arm_lengths = [len(a) == 0 for a in self.reward_arms]
        if np.any(arm_lengths):
            return [i for i in range(len(arm_lengths)) if arm_lengths[i]][0]
        reward_samples = [a.sample() for a in self.reward_arms]
        cost_samples = [a.sample() for a in self.cost_arms]
        reward_cost_ratio = np.array(reward_samples) / np.array(cost_samples)
        return int(np.argmax(reward_cost_ratio))

    def set(self, arm: int, alpha_r: float, beta_r: float, alpha_c: float, beta_c: float):
        self.reward_arms[arm].set(alpha_r, beta_r)
        self.cost_arms[arm].set(alpha_c, beta_c)

    def update(self, arm: int, reward: float, cost: float):
        if not (reward == 1 or reward == 0):
            reward = int(self.rng.uniform() < reward)
        if cost == 1 or cost == 0:
            cost = int(self.rng.uniform() < cost)
        [(r.update(reward, was_pulled=arm == i), c.update(cost, was_pulled=arm == i))
         for (i, (r, c)) in enumerate(zip(self.reward_arms, self.cost_arms))]

    def __len__(self):
        return len(self.cost_arms)

    def startup_complete(self):
        return not np.any([not a.startup_complete for a in self.cost_arms])
