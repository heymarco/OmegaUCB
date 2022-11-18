import numpy as np
from scipy.stats import norm

from components.bandits.abstract import AbstractArm, AbstractBandit


class WUCBArm(AbstractArm):
    def __init__(self, alpha=None, confidence=0.95, r=4.0, adaptive=False):
        self.r = r
        self.adaptive = adaptive
        self.confidence = confidence
        self.z = norm.interval(self.confidence)[1]
        self.pulls = 0
        self.t = 0
        self._prev_pulls = 0
        self._avg_cost = 0
        self._avg_reward = 0
        self._type = type
        self._rew = 0
        self._cost = 0
        if alpha is not None:
            self.alpha = alpha

    def _adaptive_z(self):
        if self.t == 0:
            return 0
        else:
            z = np.sqrt(2 * self.r * np.log(self.t + 1))
        return z

    def update_cost(self):
        n = self.pulls
        ns = self._avg_cost * n
        nf = n - ns
        if self.adaptive:
            z = self._adaptive_z()
        else:
            z = self.z
        z2 = z ** 2
        avg = (ns + 0.5 * z2) / (n + z2)
        ci = z / (n + z2) * np.sqrt((ns * nf) / n + z2 / 4)
        self._cost = avg - ci

    def update_reward(self):
        n = self.pulls
        ns = self._avg_reward * n
        nf = n - ns
        if self.adaptive:
            z = self._adaptive_z()
        else:
            z = self.z
        z2 = np.power(z, 2)
        avg = (ns + 0.5 * z2) / (n + z2)
        ci = z / (n + z2) * np.sqrt((ns * nf) / n + z2 / 4)
        self._rew = min(avg + ci, 1.0)

    def sample(self):
        rew = self._rew
        cost = self._cost
        if cost == 0:
            # we have not yet payed anything for this arm
            return np.infty
        return rew / cost

    def set(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, new_cost, new_reward, was_pulled):
        self.t += 1
        self._prev_pulls = self.pulls
        if self.pulls == 0 and not was_pulled:
            return
        if self.adaptive:
            if was_pulled:
                self.pulls += 1
                self._avg_cost = ((self.pulls - 1) * self._avg_cost + new_cost) / self.pulls
                self._avg_reward = ((self.pulls - 1) * self._avg_reward + new_reward) / self.pulls
            self.update_cost()
            self.update_reward()
        else:
            if was_pulled:
                self.pulls += 1
                self._avg_cost = ((self.pulls - 1) * self._avg_cost + new_cost) / self.pulls
                self._avg_reward = ((self.pulls - 1) * self._avg_reward + new_reward) / self.pulls
                self.update_cost()
                self.update_reward()

    def startup_complete(self):
        return self.pulls > 0


class WUCB(AbstractBandit):
    def __init__(self, k: int, name: str, seed: int, r: float = 4.0, adaptive: bool = False):
        super(WUCB, self).__init__(k, name, seed)
        self.r = r
        self.adaptive = adaptive
        self._startup_complete = False
        self.arms = [WUCBArm(r=r, adaptive=adaptive) for _ in range(k)]

    def sample(self):
        if not self._startup_complete:
            result = [i for i, a in enumerate(self.arms) if not a.startup_complete()][0]
            return result
        samples = [a.sample() for a in self.arms]
        return np.random.choice(
            np.flatnonzero(samples == np.max(samples))
        )

    def update(self, arm: int, reward: float, cost: float):
        [a.update(new_reward=reward, new_cost=cost, was_pulled=arm == i)
         for i, a in enumerate(self.arms)]
        if not self._startup_complete:
            self._startup_complete = self.startup_complete()

    def set(self, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return len(self.arms)

    def startup_complete(self):
        return np.alltrue([a.startup_complete() for a in self.arms])