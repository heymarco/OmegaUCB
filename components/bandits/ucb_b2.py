import numpy as np
from scipy.stats import norm

from components.bandits.abstract import AbstractArm, AbstractBandit


class Aggregate:
    def __init__(self):
        self._count = 0
        self._mean = 0.0
        self._m2 = 0

    def update(self, newval: float):
        self._count += 1
        delta = newval - self._mean
        self._mean += delta / self._count
        delta2 = newval - self._mean
        self._m2 += delta * delta2

    def variance(self):
        if self._count < 2:
            return 1 / 4
        return self._m2 / (self._count - 1)

    def mean(self):
        return self._mean

    def std(self):
        return np.sqrt(self.variance())


def compute_eta(mean: float, variance: float, n: int, m=0, M=1, min_samples=30) -> float:
    if n < min_samples:
        return 1.0
    bernoulli_variance = (M - mean) * (mean - m)
    bernoulli_sample_variance = n / (n - 1) * bernoulli_variance
    if bernoulli_variance == 0:
        return 1.0
    eta = variance / bernoulli_sample_variance
    return min(eta, 1.0)


class UCBB2Arm(AbstractArm):
    def __init__(self, alpha=None, confidence=0.95, min_samples=30):
        self.confidence = confidence
        self.min_samples = min_samples
        self.z = norm.interval(self.confidence)[1]
        self.pulls = 0
        self.t = 0
        self._prev_pulls = 0
        self._rew_aggregate = Aggregate()
        self._cost_aggregate = Aggregate()
        self._type = type
        self._eta = 0
        self._epsilon = 0
        if alpha is not None:
            self.alpha = alpha

    def update_epsilon(self):
        variance = self._rew_aggregate.variance()
        n = self.pulls

        root = np.sqrt(2 * variance * np.log(self.t ** self.alpha) / n)
        addition = 3 * 1 * np.log(self.t ** self.alpha) / n
        self._epsilon = root + addition

    def update_eta(self):
        variance = self._cost_aggregate.variance()
        n = self.pulls

        root = np.sqrt(2 * variance * np.log(self.t ** self.alpha) / n)
        addition = 3 * 1 * np.log(self.t ** self.alpha) / n
        self._eta = root + addition

    def sample(self, c_min: float):
        eta = self._eta
        eps = self._epsilon

        r = max(0.0, self._rew_aggregate.mean()) / max(c_min, self._cost_aggregate.mean())
        if eta < self._rew_aggregate.mean():  # self.check_condition_7(eta):
            c = 1.4 * (eps + r * eta) / max(self._cost_aggregate.mean(), 1e-10)
        else:
            c = np.infty
        return r + c

    def check_condition_7(self, eta, lam=1.28) -> bool:
        return 0 < eta < self._cost_aggregate.mean() * (lam - 1) / lam

    def set(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, new_cost, new_reward, was_pulled):
        self.t += 1
        self._prev_pulls = self.pulls
        if self.pulls == 0 and not was_pulled:
            return
        if was_pulled:
            self.pulls += 1
            self._cost_aggregate.update(new_cost)
            self._rew_aggregate.update(new_reward)
        self.update_eta()
        self.update_epsilon()

    def startup_complete(self):
        return self.pulls > 0


class UCBB2(AbstractBandit):
    def __init__(self, k: int, name: str, seed: int, min_samples: int = 30, alpha=2.01):
        super(UCBB2, self).__init__(k, name, seed)
        self.min_samples = min_samples
        self._startup_complete = False
        self.arms = [UCBB2Arm(min_samples=min_samples, alpha=alpha) for _ in range(k)]

    def sample(self, c_min: float):
        if not self._startup_complete:
            result = [i for i, a in enumerate(self.arms) if not a.startup_complete()][0]
            return result
        samples = [a.sample(c_min) for a in self.arms]
        return self.rng.choice(
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
