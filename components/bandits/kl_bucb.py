import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import newton, minimize

from components.bandits.abstract import AbstractArm, AbstractBandit


def optimization_function(x, p, t, n):
    if x < 0:
        return np.infty
    if x > 1:
        return np.infty
    def f(_t):
        return 1 + _t * np.log(_t) ** 2

    def f2(_t):
        return _t ** 2 / 2

    def entr(_p, _q):
        if _p == 0:
            _p += 1e-5
        if _p == 1:
            _p -= 1e-5
        if _q == 0:
            _q += 1e-5
        if _q == 1:
            _q -= 1e-5
        return _p * np.log(_p / _q) + (1 - _p) * np.log((1 - _p) / (1 - _q))

    diff = entr(p, x) - np.log(f(t)) / n
    return diff


class KLArm(AbstractArm):
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self._cost_avg = 0
        self._reward_avg = 0
        self.t = 0
        self.pulls = 0

    def sample(self):
        reward = self._upper_ci(self._reward_avg)
        cost = self._lower_ci(self._cost_avg)
        # incorporate alpha:
        reward = self._reward_avg + self.alpha * (reward - self._reward_avg)
        cost = self._cost_avg - self.alpha * (self._cost_avg - cost)
        reward = min(1.0, reward)
        cost = max(1e-10, cost)
        return reward / cost

    def set(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, new_reward, new_cost, was_pulled):
        self.t += 1
        if self.pulls == 0 and not was_pulled:
            return
        if was_pulled:
            self.pulls += 1
            self._cost_avg = ((self.pulls - 1) * self._cost_avg + new_cost) / self.pulls
            self._reward_avg = ((self.pulls - 1) * self._reward_avg + new_reward) / self.pulls

    def _lower_ci(self, avg: float):
        if avg == 0:
            return avg
        mus = newton(lambda x: optimization_function(x, avg, self.t, self.pulls),
                     x0=0.01)
        mu_low = np.min(mus)
        return mu_low

    def _upper_ci(self, avg: float):
        if avg == 1:
            return avg
        mus = newton(lambda x: optimization_function(x, avg, self.t, self.pulls),
                     x0=0.99)
        mu_high = np.min(mus)
        return mu_high

    def startup_complete(self):
        return self.pulls > 0


class KLBUCB(AbstractBandit):
    def __init__(self, k: int, name: str, seed: int):
        super().__init__(k, name, seed)
        self.arms = [KLArm() for _ in range(k)]

    def sample(self):
        startup_complete = [a.startup_complete() for a in self.arms]
        if not np.alltrue(startup_complete):
            return [i for i in range(self.k) if not startup_complete[i]][0]
        samples = [a.sample() for a in self.arms]
        return np.argmax(samples)

    def update(self, arm: int, reward: float, cost: float):
        [a.update(new_reward=reward, new_cost=cost, was_pulled=arm == i)
         for i, a in enumerate(self.arms)]

    def set(self, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return len(self.arms)


if __name__ == '__main__':
    x = np.arange(1, 1000) / 1000
    dummy_arm = KLArm()
    dummy_arm._cost_avg = 0.05
    dummy_arm._reward_avg = 0.52
    dummy_arm.t = 1000
    dummy_arm.pulls = 300
    y_reward = [optimization_function(_x,
                                      p=dummy_arm._reward_avg,
                                      t=dummy_arm.t,
                                      n=dummy_arm.pulls)
                for _x in x]
    y_cost = [optimization_function(_x,
                                    p=dummy_arm._cost_avg,
                                    t=dummy_arm.t,
                                    n=dummy_arm.pulls)
              for _x in x]
    reward = dummy_arm._upper_ci(dummy_arm._reward_avg)
    cost = dummy_arm._lower_ci(dummy_arm._cost_avg)
    reward = dummy_arm._reward_avg + dummy_arm.alpha * (reward - dummy_arm._reward_avg)
    cost = dummy_arm._cost_avg - dummy_arm.alpha * (dummy_arm._cost_avg - cost)
    reward = min(1.0, reward)
    cost = max(1e-10, cost)
    plt.plot(x, y_reward)
    plt.plot(x, y_cost)
    plt.axhline(0)
    plt.axvline(reward, color="orange")
    plt.axvline(cost, color="blue")
    plt.show()
