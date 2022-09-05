import numpy as np

from components.bandits.abstract import AbstractBandit, AbstractArm


class UCBMBArm(AbstractArm):
    def __init__(self, is_reward_arm: bool, c_min=1 / 100):
        self.is_reward_arm = is_reward_arm
        self.t = 0
        self.avg = 0
        self.pulls = 0
        self.c_min = c_min
        self.startup_complete = False

    def e(self):
        if self.pulls <= 1:
            return 0.0
        root = np.sqrt(2 * np.log(self.t) / self.pulls)
        _e = (root * (1 + 1 / self.c_min)) / (self.c_min - root)
        return _e

    def sample(self):
        if self.is_reward_arm:
            return self.avg
        else:
            return max(self.avg, self.c_min)

    def set(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, new_val, was_pulled: bool):
        self.t += 1
        if was_pulled:
            self.startup_complete = True
            self.avg = (self.pulls * self.avg + new_val) / (self.pulls + 1)
            self.pulls += 1


class UCBMBBandit(AbstractBandit):
    def __init__(self, k, name: str, seed: int, kappa: float = 2e-4):
        super().__init__(k, name, seed)
        self.reward_arms = [UCBMBArm(is_reward_arm=True) for _ in range(k)]
        self.cost_arms = [UCBMBArm(is_reward_arm=False) for _ in range(k)]

    def sample(self):
        if not self._startup_complete():
            return next(i for i in range(self.k) if not self.reward_arms[i].startup_complete)
        else:
            rewards = np.array([a.sample() for a in self.reward_arms])
            costs = np.array([a.sample() for a in self.cost_arms])
            e = np.array([a.e() for a in self.reward_arms])  # is the same for the cost arms.
            ratio = rewards / costs + e
            return np.argmax(ratio)

    def update(self, arm: int, reward: float, cost: float):
        if not (reward == 1 or reward == 0):
            reward = int(self.rng.uniform() < reward)
        if cost == 1 or cost == 0:
            cost = int(self.rng.uniform() < cost)
        [self.reward_arms[i].update(reward, was_pulled=arm == i) for i in range(self.k)]
        [self.cost_arms[i].update(cost, was_pulled=arm == i) for i in range(self.k)]

    def set(self, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return self.k

    def _startup_complete(self):
        startup_complete = [a.startup_complete for a in self.reward_arms + self.cost_arms]
        return np.alltrue(startup_complete)