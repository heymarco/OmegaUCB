import numpy as np

from components.bandits.abstract import AbstractArm, AbstractBandit


class ArmWithBetaPosterior(AbstractArm):
    def __init__(self, seed: int):
        self.alpha = 0.0
        self.beta = 0.0
        self.rng = np.random.default_rng(seed)

    def sample(self):
        return self.rng.beta(a=self.alpha + 1, b=self.beta + 1)

    def mean(self):
        return (self.alpha + 1) / (self.alpha + self.beta + 2)

    def set(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    def update(self, new_val: float):
        self.alpha += new_val
        self.beta += 1 - new_val


class BudgetedThompsonSampling(AbstractBandit):
    def __init__(self, k: int, name: str, seed: int):
        self.reward_arms = [ArmWithBetaPosterior(seed + arm_index) for arm_index in range(k)]
        self.cost_arms = [ArmWithBetaPosterior(seed + k + arm_index) for arm_index in range(k)]
        super(BudgetedThompsonSampling, self).__init__(k, name, seed)

    def sample(self) -> int:
        reward_samples = [a.sample() for a in self.reward_arms]
        cost_samples = [a.sample() for a in self.cost_arms]
        reward_cost_ratio = np.array(reward_samples) / np.array(cost_samples)
        return int(np.argmax(reward_cost_ratio))

    def set(self, arm: int, alpha_r: float, beta_r: float, alpha_c: float, beta_c: float):
        self.reward_arms[arm].set(alpha_r, beta_r)
        self.cost_arms[arm].set(alpha_c, beta_c)

    def update(self, arm: int, reward: float, cost: float):
        if reward == 1 or reward == 0:
            # Bernoulli reward
            self.reward_arms[arm].update(reward)
        else:
            bernoulli_reward = int(self.rng.uniform() < reward)
            self.reward_arms[arm].update(bernoulli_reward)
        if cost == 1 or cost == 0:
            # Bernoulli cost
            self.cost_arms[arm].update(cost)
        else:
            bernoulli_cost = int(self.rng.uniform() < cost)
            self.cost_arms[arm].update(bernoulli_cost)

    def __len__(self):
        return len(self.cost_arms)