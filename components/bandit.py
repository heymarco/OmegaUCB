import numpy as np
from scipy.stats import beta

from components.bandits.abstract import AbstractArm, AbstractBandit


class ArmWithAdaptiveBetaPosterior(AbstractArm):
    def __init__(self, seed: int, ci: str, is_cost_arm: bool):
        self.is_cost_arm = is_cost_arm
        self.rng = np.random.default_rng(seed)
        self.alpha = 0.0
        self.beta = 0.0
        self.startup_complete = False
        self.prev_avg = np.nan
        self.this_avg = np.nan
        self.pulls = 0
        self.type = ci
        self.t = 0

    def __len__(self):
        return self.pulls

    def exp_decay_alpha(self, alpha_max=0.1, k=0.01):
        alpha = alpha_max * (1 - np.exp(-k * (self.t - 1)))
        return max(1e-5, min(1 - 1e-5, alpha))  # avoid singularities

    def sample(self):
        mean = (self.alpha + 1) / (self.alpha + self.beta + 2)
        s = self.rng.beta(a=self.alpha + 1, b=self.beta + 1)
        if self.type == "optimistic":
            if self.is_cost_arm and s > mean:
                while s > mean:
                    s = self.rng.beta(a=self.alpha + 1, b=self.beta + 1)
            elif (not self.is_cost_arm) and s < mean:
                while s < mean:
                    s = self.rng.beta(a=self.alpha + 1, b=self.beta + 1)
        elif self.type == "pessimistic":
            if self.is_cost_arm and s < mean:
                while s < mean:
                    s = self.rng.beta(a=self.alpha + 1, b=self.beta + 1)
            elif (not self.is_cost_arm) and s > mean:
                while s > mean:
                    s = self.rng.beta(a=self.alpha + 1, b=self.beta + 1)
        elif self.type == "ts-cost":
            if self.is_cost_arm:
                s = self.rng.beta(a=self.alpha + 1, b=self.beta + 1)
            else:
                s = (self.alpha + 1) / (self.alpha + self.beta + 2)
        elif self.type == "ts-reward":
            if not self.is_cost_arm:
                s = self.rng.beta(a=self.alpha + 1, b=self.beta + 1)
            else:
                s = (self.alpha + 1) / (self.alpha + self.beta + 2)
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

    def jeffrey(self, alpha=0.05):
        n = self.pulls
        x = self.this_avg * n
        low, high = beta.interval(alpha=alpha, a=0.5 + x, b=0.5 + n - x)
        return low if self.is_cost_arm else high

    def _compute_alpha_beta(self):
        alpha = self.this_avg * self.pulls
        beta = self.pulls * (1 - self.this_avg)
        return alpha, beta


class AdaptiveBudgetedThompsonSampling(AbstractBandit):
    def __init__(self, k: int, name: str, seed: int, ci_reward: str, ci_cost: str):
        self.ci_reward = ci_reward
        self.ci_cost = ci_cost
        self.reward_arms = [ArmWithAdaptiveBetaPosterior(arm_index, ci=ci_reward, is_cost_arm=False) for arm_index in range(k)]
        self.cost_arms = [ArmWithAdaptiveBetaPosterior(k + arm_index, ci=ci_cost, is_cost_arm=True) for arm_index in range(k)]
        self._startup_complete = False
        super(AdaptiveBudgetedThompsonSampling, self).__init__(k, name, seed)

    def sample(self) -> int:
        if not self._startup_complete:
            return [i for i, a in enumerate(self.reward_arms) if not a.startup_complete][0]
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
        if not self._startup_complete:
            self._startup_complete = np.any([not a.startup_complete for a in self.reward_arms])

    def __len__(self):
        return len(self.cost_arms)

    def startup_complete(self):
        return not np.any([not a.startup_complete for a in self.cost_arms])
