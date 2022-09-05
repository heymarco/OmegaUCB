import numpy as np

from components.bandits.abstract import AbstractArm, AbstractBandit


class UCBArm(AbstractArm):
    def __init__(self, type: str, alpha=0.25):
        self.alpha = alpha
        self.pulls = 0
        self.t = 0
        self._avg_cost = np.nan
        self._avg_reward = np.nan
        self._type = type

    def _epsilon(self):
        return self.alpha * np.log(self.t - 1) / self.pulls

    def sample(self):
        if self._type == "i":
            return self._avg_reward / self._avg_cost + self._epsilon()
        elif self._type == "c":
            return (self._avg_reward + self._epsilon()) / self._avg_cost
        elif self._type == "m":
            top = min(self._avg_reward + self._epsilon(), 1)
            bottom = max(self._avg_cost - self._epsilon(), 1e-10)
            return top / bottom
        else:
            raise ValueError

    def set(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, new_cost, new_reward, was_pulled):
        self.t += 1
        self.pulls += 1 if was_pulled else self.pulls
        self._avg_cost = ((self.pulls - 1) * self._avg_cost + new_cost) / self.pulls
        self._avg_reward = ((self.pulls - 1) * self._avg_reward + new_cost) / self.pulls


class UCB(AbstractBandit):
    def __init__(self, k: int, name: str, type: str, seed: int):
        super().__init__(k, name, seed)
        self.type = type
        self.arms = [UCBArm(type) for _ in range(k)]

    def sample(self):
        samples = [a.sample() for a in self.arms]
        return np.argmax(samples)

    def update(self, arm: int, reward: float, cost: float):
        [a.update(new_reward=reward, new_cost=cost, was_pulled=arm == i)
         for i, a in enumerate(self.arms)]

    def set(self, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return len(self.arms)