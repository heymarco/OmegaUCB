import numpy as np
from scipy import stats

from components.bandits.abstract import AbstractArm, AbstractBandit
from components.bandits.thompson import ArmWithBetaPosterior


class EpsilonArmWithBetaPosterior(AbstractArm):
    def __init__(self, seed: int, var_tolerance: float = 10):
        self.var_tolerance = var_tolerance
        self.rng = np.random.default_rng(seed)
        self.startup_complete = False
        self.prev_reward_avg = np.nan
        self.reward_avg = np.nan
        self.prev_cost_avg = np.nan
        self.cost_avg = np.nan
        self.pulls = 0
        self.t = 0
        self.reward_arm = ArmWithBetaPosterior(seed=seed)
        self.cost_arm = ArmWithBetaPosterior(seed=seed+1)

    def __len__(self):
        return self.pulls

    def variance(self):
        mu_a = self.reward_arm.mean()
        mu_b = self.cost_arm.mean()
        s2_a = self.reward_arm.variance()
        s2_b = self.cost_arm.variance()
        return mu_a ** 2 / mu_b ** 2 * (s2_a / mu_a ** 2 + s2_b / mu_b ** 2)
        # return 1 / self.cost_arm.mean() ** 2 * ( self.reward_arm.variance() + self.reward_arm.mean() ** 2 * self.cost_arm.variance())

    def compute_epsilon(self):
        a = self.cost_arm.mean() ** 2
        b = self.reward_arm.variance()
        c = self.cost_arm.variance()
        d = self.reward_arm.mean() ** 2
        K = self.var_tolerance

        root = np.sqrt((b ** 2 + 4 * c * d * K) / K ** 2)
        # eps = (b - K * (2 * a + root)) / (2 * K)
        eps = (-2 * a * K + K * root + b) / (2 * K)
        return eps

    def sample(self):
        rew = self.reward_arm.sample()
        cost = self.cost_arm.sample()
        return rew / cost

    def set(self, alpha: float, beta: float):
        raise NotImplementedError

    def update(self, new_reward: float, new_cost: float, was_pulled: bool):
        self.t += 1
        if was_pulled:
            if not self.startup_complete:
                self.prev_reward_avg = new_reward
                self.prev_cost_avg = new_cost
                self.reward_avg = new_reward
                self.cost_avg = new_cost
                self.pulls += 1
                self.startup_complete = True
            else:
                self.prev_reward_avg = self.reward_avg
                new_reward = (self.pulls * self.prev_reward_avg + new_reward) / (self.pulls + 1)
                self.reward_avg = new_reward
                self.prev_cost_avg = self.cost_avg
                new_cost = (self.pulls * self.prev_cost_avg + new_cost) / (self.pulls + 1)
                self.cost_avg = new_cost
                self.pulls += 1
        if self.startup_complete:
            alpha_r = self.reward_avg * self.pulls
            beta_r = self.pulls - alpha_r
            self.reward_arm.set(alpha_r, beta_r)
            eps = self.compute_epsilon()
            alpha_c = self.cost_avg * self.pulls
            beta_c = self.pulls - alpha_c

            def alpha_addend(e: float):
                a = alpha_c + 1
                b = beta_c + 1
                c = a / (a + b)
                x = np.sqrt((a + b + 2) ** 2 * (c + e)) - a - 1
                return x
            eps = alpha_addend(eps)
            eps = max(0, eps)
            alpha_c = max(0.0, alpha_c + eps)
            beta_c = max(0.0, beta_c - eps)
            self.cost_arm.set(alpha_c, beta_c)



class EpsilonBudgetedThompsonSampling(AbstractBandit):
    def __init__(self, k: int, name: str, seed: int, var_tolerance = 10.0):
        self.arms = [EpsilonArmWithBetaPosterior(arm_index, var_tolerance) for arm_index in range(k)]
        super(EpsilonBudgetedThompsonSampling, self).__init__(k, name, seed)

    def sample(self) -> int:
        if np.any([not a.startup_complete for a in self.arms]):
            return [i for i, a in enumerate(self.arms) if not a.startup_complete][0]
        return int(np.argmax([a.sample() for a in self.arms]))

    def set(self, arm: int, alpha_r: float, beta_r: float, alpha_c: float, beta_c: float):
        raise NotImplementedError

    def update(self, arm: int, reward: float, cost: float):
        if not (reward == 1 or reward == 0):
            reward = int(self.rng.uniform() < reward)
        if cost == 1 or cost == 0:
            cost = int(self.rng.uniform() < cost)
        [a.update(reward, cost, was_pulled=arm == i) for (i, a) in enumerate(self.arms)]

    def __len__(self):
        return len(self.arms)

    def startup_complete(self):
        return not np.any([not a.startup_complete for a in self.arms])


if __name__ == '__main__':
    rewards = np.random.uniform(size=300) < 0.5
    cost_continuous = np.random.uniform(size=300)
    for cost in range(1, 2):
        bandit = EpsilonBudgetedThompsonSampling(k=1, name="EBTS", seed=0, var_tolerance=3)
        costs = cost_continuous < cost / 200
        for i, (r, c) in enumerate(zip(rewards, costs)):
            bandit.update(arm=0, reward=r, cost=c)
            print(bandit.arms[0].reward_arm.variance())
            print(i, bandit.arms[0].compute_epsilon(), bandit.arms[0].variance())

