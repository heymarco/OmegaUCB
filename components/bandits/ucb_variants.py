import numpy as np
from scipy import stats

from components.bandits.abstract import AbstractArm, AbstractBandit


class UCBArm(AbstractArm):
    def __init__(self, type: str, alpha=0.25, confidence=0.95, adaptive=False):
        self.adaptive = adaptive
        self.confidence = confidence
        self.alpha = alpha
        self.pulls = 0
        self.t = 0
        self._prev_pulls = 0
        self._avg_cost = 0
        self._avg_reward = 0
        self._type = type
        self._rew = 0
        self._cost = 0

    def _wilson_alpha(self):
        return 1 - self.confidence

    def _adaptive_confidence(self):
        conf = max(0, 1 - 2 / self.t ** 2)
        return conf

    def _adaptive_z(self):
        z = stats.norm.interval(self._adaptive_confidence())[1]
        return z

    def _epsilon(self):
        return self.alpha * np.sqrt(np.log(self.t - 1) / self.pulls)

    def update_wilson_estimate_cost(self):
        n = self.pulls
        ns = self._avg_cost * n
        nf = n - ns
        if self.adaptive:
            z = self._adaptive_z()
        else:
            z = stats.norm.interval(self.confidence)[1]
        z2 = z ** 2

        avg = (ns + 0.5 * z2) / (n + z2)
        ci = z / (n + z2) * np.sqrt((ns * nf) / n + z2 / 4)
        self._cost = max(avg - ci, 1e-10)

    def update_wilson_estimate_reward(self):
        n = self.pulls
        ns = self._avg_reward * n
        nf = n - ns
        if self.adaptive:
            z = self._adaptive_z()
        else:
            z = stats.norm.interval(self.confidence)[1]
        z2 = np.power(z, 2)
        avg = (ns + 0.5 * z2) / (n + z2)
        ci = z / (n + z2) * np.sqrt((ns * nf) / n + z2 / 4)
        self._rew = min(avg + ci, 1.0)

    def update_jeffrey_estimate_cost(self):
        if self.pulls == self._prev_pulls and self.pulls > 1:
            return self._cost
        n = self.pulls
        x = self._avg_cost * n
        if self.adaptive:
            low, high = stats.beta.interval(alpha=self._adaptive_confidence(),
                                            a=0.5 + x, b=0.5 + n - x)
        else:
            low, high = stats.beta.interval(alpha=self.confidence,
                                            a=0.5 + x, b=0.5 + n - x)
        self._cost = low

    def update_jeffrey_estimate_reward(self):
        if self.pulls == self._prev_pulls and self.pulls > 1:
            return self._rew
        n = self.pulls
        x = self._avg_reward * n
        if self.adaptive:
            low, high = stats.beta.interval(alpha=self._adaptive_confidence(),
                                            a=0.5 + x, b=0.5 + n - x)
        else:
            low, high = stats.beta.interval(alpha=self.confidence,
                                            a=0.5 + x, b=0.5 + n - x)
        self._rew = high

    def sample(self):
        cost = max(1e-10, self._avg_cost)
        if self._type == "i":
            return self._avg_reward / cost + self._epsilon()
        elif self._type == "c":
            return (self._avg_reward + self._epsilon()) / cost
        elif self._type == "m":
            top = min(self._avg_reward + self._epsilon(), 1)
            bottom = max(cost - self._epsilon(), 1e-10)
            return top / bottom
        elif self._type == "j" or self._type == "w":
            rew = self._rew
            cost = self._cost
            return rew / cost
        else:
            raise ValueError

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
            if self._type == "j":
                self.update_jeffrey_estimate_cost()
                self.update_jeffrey_estimate_reward()
            elif self._type == "w":
                self.update_wilson_estimate_cost()
                self.update_wilson_estimate_reward()
        else:
            if was_pulled:
                self.pulls += 1
                self._avg_cost = ((self.pulls - 1) * self._avg_cost + new_cost) / self.pulls
                self._avg_reward = ((self.pulls - 1) * self._avg_reward + new_reward) / self.pulls
                if self._type == "j":
                    self.update_jeffrey_estimate_cost()
                    self.update_jeffrey_estimate_reward()
                elif self._type == "w":
                    self.update_wilson_estimate_cost()
                    self.update_wilson_estimate_reward()

    def startup_complete(self):
        return self.pulls > 0


class UCB(AbstractBandit):
    def __init__(self, k: int, name: str, type: str, seed: int, adaptive: bool = False):
        super().__init__(k, name, seed)
        self.type = type
        self.adaptive = adaptive
        self.arms = [UCBArm(type, adaptive=adaptive) for _ in range(k)]

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
