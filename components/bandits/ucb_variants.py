import numpy as np
from scipy import stats

from components.bandits.abstract import AbstractArm, AbstractBandit


class UCBArm(AbstractArm):
    def __init__(self, type: str, alpha=0.25, alpha_wilson=0.05):
        self.alpha = alpha
        self.wilson_alpha = alpha_wilson
        self.pulls = 0
        self.t = 0
        self._avg_cost = 0
        self._avg_reward = 0
        self._type = type
        self._cmin = 1e-5

    def _epsilon(self):
        return self.alpha * np.sqrt(np.log(self.t - 1) / self.pulls)

    def _wilson_reward_estimate(self):
        ns = self._avg_reward * self.pulls
        n = self.pulls
        z = 1.96 if self.wilson_alpha == 0.05 else stats.norm.interval(1 - self.wilson_alpha)
        z2 = z ** 2
        return (ns + 0.5 * z2) / (n + z2)

    def _wilson_cost_estimate(self):
        ns = self._avg_cost * self.pulls
        n = self.pulls
        z = 1.96 if self.wilson_alpha == 0.05 else stats.norm.interval(1 - self.wilson_alpha)
        z2 = z ** 2
        return (ns + 0.5 * z2) / (n + z2)

    def _wilson_reward_ci(self):
            n = self.pulls
            ns = self._avg_reward * n
            nf = n - ns
            z = 1.96 if self.wilson_alpha == 0.05 else stats.norm.interval(1 - self.wilson_alpha)[1]
            z2 = np.power(z, 2)
            return 2 / (n + z2) * np.sqrt((ns * nf) / n + z2 / 4)

    def _wilson_cost_ci(self):
        n = self.pulls
        ns = self._avg_cost * n
        nf = n - ns
        z = 1.96 if self.wilson_alpha == 0.05 else stats.norm.interval(1 - self.wilson_alpha)[1]
        z2 = np.power(z, 2)
        return 2 / (n + z2) * np.sqrt((ns * nf) / n + z2 / 4)

    def sample(self):
        cost = max(self._cmin, self._avg_cost)
        if self._type == "i":
            return self._avg_reward / cost + self._epsilon()
        elif self._type == "c":
            return (self._avg_reward + self._epsilon()) / cost
        elif self._type == "m":
            top = min(self._avg_reward + self._epsilon(), 1)
            bottom = max(cost - self._epsilon(), 1e-10)
            return top / bottom
        elif self._type == "w":
            ci_rew = self._wilson_reward_ci()
            ci_cost = self._wilson_cost_ci()
            top = min(self._wilson_reward_estimate() + ci_rew, 1)
            bottom = max(self._wilson_cost_estimate() - ci_cost, 1e-10)
            return top / bottom
        else:
            raise ValueError

    def set(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, new_cost, new_reward, was_pulled):
        self.t += 1
        if self.pulls == 0 and not was_pulled:
            return
        if was_pulled:
            self.pulls += 1
            self._avg_cost = ((self.pulls - 1) * self._avg_cost + new_cost) / self.pulls
            self._avg_reward = ((self.pulls - 1) * self._avg_reward + new_reward) / self.pulls

    def startup_complete(self):
        return self.pulls > 0


class UCB(AbstractBandit):
    def __init__(self, k: int, name: str, type: str, seed: int):
        super().__init__(k, name, seed)
        self.type = type
        self.arms = [UCBArm(type) for _ in range(k)]

    def sample(self):
        if not self.startup_complete():
            return [i for i, a in enumerate(self.arms) if not a.startup_complete()][0]
        samples = [a.sample() for a in self.arms]
        return np.argmax(samples)

    def update(self, arm: int, reward: float, cost: float):
        [a.update(new_reward=reward, new_cost=cost, was_pulled=arm == i)
         for i, a in enumerate(self.arms)]

    def set(self, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return len(self.arms)

    def startup_complete(self):
        return np.alltrue([a.startup_complete() for a in self.arms])