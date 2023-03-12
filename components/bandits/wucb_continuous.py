import numpy as np
from scipy.stats import norm

from components.bandits.abstract import AbstractArm, AbstractBandit


class Aggregate:
    def __init__(self):
        self._count = 0
        self._mean = 0
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


class GeneralizedWUCBArm(AbstractArm):
    def __init__(self, alpha=None, confidence=0.95, r=4.0, adaptive=False, min_samples=30):
        self.rho = r
        self.adaptive = adaptive
        self.confidence = confidence
        self.min_samples = min_samples
        self.z = norm.interval(self.confidence)[1]
        self.pulls = 0
        self.t = 0
        self._prev_pulls = 0
        self._rew_aggregate = Aggregate()
        self._cost_aggregate = Aggregate()
        self._type = type
        self._rew = 0
        self._cost = 0
        if alpha is not None:
            self.alpha = alpha

    def _adaptive_z(self):
        if self.t == 0:
            return 0
        else:
            z = np.sqrt(2 * self.rho * np.log(self.t + 1))
        return z

    def update_cost(self):
        mean = self._cost_aggregate.mean()
        n = self.pulls
        ns = mean * n
        nf = n - ns
        eta = compute_eta(mean, self._cost_aggregate.variance(), n=n, min_samples=self.min_samples)
        if self.adaptive:
            z = self._adaptive_z()
        else:
            z = self.z
        z2 = z ** 2 * eta
        z = np.sqrt(z2)
        avg = (ns + 0.5 * z2) / (n + z2)
        ci = z / (n + z2) * np.sqrt((ns * nf) / n + z2 / 4)
        self._cost = avg - ci

    def update_reward(self):
        mean = self._rew_aggregate.mean()
        n = self.pulls
        ns = mean * n
        nf = n - ns
        eta = compute_eta(mean, self._rew_aggregate.variance(), n=n, min_samples=self.min_samples)
        if self.adaptive:
            z = self._adaptive_z()
        else:
            z = self.z
        z2 = z ** 2 * eta
        z = np.sqrt(z2)
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
                self._cost_aggregate.update(new_cost)
                self._rew_aggregate.update(new_reward)
            self.update_cost()
            self.update_reward()
        else:
            if was_pulled:
                self.pulls += 1
                self._rew_aggregate.update(new_reward)
                self._cost_aggregate.update(new_cost)
                self.update_cost()
                self.update_reward()

    def startup_complete(self):
        return self.pulls > 0


class GeneralizedWUCB(AbstractBandit):
    def __init__(self, k: int, name: str, seed: int, r: float = 4.0, adaptive: bool = False, min_samples: int = 30):
        super(GeneralizedWUCB, self).__init__(k, name, seed)
        self.min_samples = min_samples
        self.r = r
        self.adaptive = adaptive
        self._startup_complete = False
        self.arms = [GeneralizedWUCBArm(r=r, adaptive=adaptive, min_samples=min_samples) for _ in range(k)]

    def sample(self):
        if not self._startup_complete:
            result = [i for i, a in enumerate(self.arms) if not a.startup_complete()][0]
            return result
        samples = [a.sample() for a in self.arms]
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