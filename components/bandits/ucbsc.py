import numpy as np
from scipy.stats import norm

from components.bandits.abstract import AbstractArm, AbstractBandit


class UCBSCArm(AbstractArm):
    def __init__(self):
        self.pulls = 0
        self.t = 0
        self._prev_pulls = 0
        self._avg_cost = 0
        self._avg_reward = 0
        self._rew = 0
        self._cost = 0

    def sample(self):
        mu = self._avg_reward
        tau = self._avg_cost
        aux_term = np.log(self.t / self.pulls) / (2 * self.pulls)
        if tau < np.sqrt(aux_term):
            return np.infty
        else:
            alpha = np.sqrt(aux_term / (mu ** 2 + tau ** 2 - aux_term))
            return (mu + alpha * tau) / (tau - alpha * mu)

    def set(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, new_cost, new_reward, was_pulled):
        self.t += 1
        self._prev_pulls = self.pulls
        if self.pulls == 0 and not was_pulled:
            return
        if was_pulled:
            self.pulls += 1
            self._avg_cost = ((self.pulls - 1) * self._avg_cost + new_cost) / self.pulls
            self._avg_reward = ((self.pulls - 1) * self._avg_reward + new_reward) / self.pulls

    def startup_complete(self):
        return self.pulls > 0


class UCBSC(AbstractBandit):
    def __init__(self, k: int, name: str, seed: int):
        super(UCBSC, self).__init__(k, name, seed)
        self.type = type
        self._startup_complete = False
        self.arms = [UCBSCArm() for _ in range(k)]

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
