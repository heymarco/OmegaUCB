import numpy as np

from components.bandits.abstract import AbstractBandit, AbstractArm


class MRCBArm(AbstractArm):
    def __init__(self, is_reward_arm: bool, kappa: float = 2e-4):
        self.is_reward_arm = is_reward_arm
        self.kappa = kappa
        self.avg = 0
        self.prev_avg = 0
        self.pulls_prev = 0
        self.pulls = 0
        self.t = 0
        self.startup_complete = False

    def _epsilon(self):
        return np.sqrt((self.kappa * np.log(self.t - 1)) / self.pulls_prev)

    def sample(self):
        if self.is_reward_arm:
            return min(1.0, self.prev_avg + self._epsilon())
        else:
            return max(0.0, self.prev_avg - self._epsilon())

    def set(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, new_val, was_pulled: bool):
        self.t += 1
        self.pulls_prev = self.pulls
        if was_pulled:
            self.pulls += 1
            if self.pulls == 1:
                self.pulls_prev = self.pulls
                self.avg = new_val
                self.prev_avg = new_val
                self.startup_complete = True
            else:
                self.prev_avg = self.avg
                new_avg = (self.pulls_prev * self.avg + new_val) / self.pulls
                self.avg = new_avg


class MRCBBandit(AbstractBandit):
    def __init__(self, k, name: str, seed: int, kappa: float = 2e-4):
        super().__init__(k, name, seed)
        self.reward_arms = [MRCBArm(is_reward_arm=True, kappa=kappa) for _ in range(k)]
        self.cost_arms = [MRCBArm(is_reward_arm=False, kappa=kappa) for _ in range(k)]

    def sample(self):
        if not self._startup_complete():
            return next(i for i in range(self.k) if not self.reward_arms[i].startup_complete)
        else:
            rewards = np.array([a.sample() for a in self.reward_arms])
            costs = np.array([a.sample() for a in self.cost_arms])
            if np.any(costs == 0):
                zero_cost_arms = [i for i in range(len(costs)) if costs[i] == 0]
                return self.rng.choice(zero_cost_arms)
            else:
                return np.argmax(rewards / costs)

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