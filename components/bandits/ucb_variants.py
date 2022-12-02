import numpy as np
from scipy.stats import norm

from components.bandits.abstract import AbstractArm, AbstractBandit


class UCBArm(AbstractArm):
    def __init__(self, type: str, alpha=None, confidence=0.95, adaptive=False):
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
        else:
            # based on Fig. 3 in paper https://www.sciencedirect.com/science/article/pii/S0925231217304216
            if type == "c":
                self.alpha = np.power(2.0, -3.0)
            elif type == "i":
                self.alpha = np.power(2.0, -2.0)
            elif type == "m":
                self.alpha = np.power(2.0, -4.0)  # this is the recommendation in the paper
                self.alpha = np.power(2.0, -2)  # but this works better!

    def _adaptive_confidence(self):
        conf = 1 - 2 / self.t ** 2
        conf = max(0, conf)
        return conf

    def _epsilon(self):
        return self.alpha * np.sqrt(np.log(self.t - 1) / self.pulls)

    def _hoeffding_epsilon_for_confidence(self):
        return np.sqrt(np.log(2 / (1 - self.confidence)) / (2 * self.pulls))

    def sample(self):
        cost = max(1e-10, self._avg_cost)
        if self._type == "i":
            epsilon = self._epsilon() if self.adaptive else self._hoeffding_epsilon_for_confidence()
            return self._avg_reward / cost + epsilon
        elif self._type == "c":
            epsilon = self._epsilon() if self.adaptive else self._hoeffding_epsilon_for_confidence()
            return (self._avg_reward + epsilon) / cost
        elif self._type == "m":
            epsilon = self._epsilon() if self.adaptive else self._hoeffding_epsilon_for_confidence()
            top = min(self._avg_reward + epsilon, 1)
            bottom = cost - epsilon
            if bottom <= 0:
                return np.infty
            return top / bottom
        else:
            raise ValueError

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


class UCB(AbstractBandit):
    def __init__(self, k: int, name: str, type: str, seed: int, adaptive: bool = False):
        super(UCB, self).__init__(k, name, seed)
        self.type = type
        self.adaptive = adaptive
        self._startup_complete = False
        self.arms = [UCBArm(type, adaptive=adaptive) for _ in range(k)]

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
