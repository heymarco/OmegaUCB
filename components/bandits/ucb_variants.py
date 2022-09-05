import numpy as np

from components.bandits.abstract import AbstractArm, AbstractBandit


class UCBArm(AbstractArm):
    def __init__(self, type: str, alpha=0.25):
        self.alpha = alpha
        self.pulls = 0
        self.t = 0
        self._avg_cost = 0
        self._avg_reward = 0
        self._type = type
        self._cmin = 1e-5

    def _epsilon(self):
        return self.alpha * np.sqrt(np.log(self.t - 1) / self.pulls)

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