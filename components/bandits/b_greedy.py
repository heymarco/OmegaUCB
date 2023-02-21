import numpy as np

from components.bandits.abstract import AbstractBandit


class BGreedy(AbstractBandit):
    def __init__(self, k: int, name: str, seed: int):
        super(BGreedy, self).__init__(k, name, seed)
        self._startup_complete = False
        self.k = k
        self.t = 0
        self.pulls = np.zeros(shape=self.k)
        self.cumulative_cost = np.array([0.0 for _ in range(k)])
        self.cumulative_reward = np.array([0.0 for _ in range(k)])

    def sample(self, c_min=None):
        if not self._startup_complete:
            result = next(i for i in range(self.k) if self.pulls[i] == 0)
            return result
        else:
            epsilon_t = min(1.0, self.k / self.t)
            if self.rng.uniform() < epsilon_t:
                return self.rng.choice(np.arange(self.k))
            else:
                efficiencies = self.cumulative_reward / (self.cumulative_cost + 1e-10)
                return self.rng.choice(
                    np.flatnonzero(efficiencies == np.max(efficiencies))
                )

    def update(self, arm: int, reward: float, cost: float):
        self.t += 1
        self.pulls[arm] += 1
        self.cumulative_reward[arm] += reward
        self.cumulative_cost[arm] += cost
        if not self._startup_complete:
            self._startup_complete = self.startup_complete()

    def set(self, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return self.k

    def startup_complete(self):
        return np.alltrue([self.pulls > 0])
