import numpy as np

from components.bandits.abstract import AbstractBandit
from components.bandits.bts import ArmWithBetaPosterior


class ThompsonSampling(AbstractBandit):
    def __init__(self, k: int, name: str, seed: int):
        self.arms = [ArmWithBetaPosterior(seed + arm_index) for arm_index in range(k)]
        super(ThompsonSampling, self).__init__(k, name, seed)

    def sample(self) -> int:
        samples = [a.sample() for a in self.arms]
        return int(np.argmax(samples))

    def set(self, arm: int, alpha: float, beta: float):
        self.arms[arm].set(alpha, beta)

    def update(self, arm: int, reward: float):
        if reward == 1 or reward == 0:
            # Bernoulli reward
            self.arms[arm].update(reward)
        else:
            bernoulli_reward = int(self.rng.uniform() < reward)
            self.arms[arm].update(bernoulli_reward)

    def alphas(self):
        return np.array([a.alpha for a in self.arms])

    def betas(self):
        return np.array([a.beta for a in self.arms])

    def __len__(self):
        return len(self.arms)