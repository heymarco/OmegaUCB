from typing import Tuple

from components.experiments.abstract import Environment


class BernoulliSamplingEnvironment(Environment):
    def sample(self, arm_index: int) -> Tuple[int, int, float, float]:
        mean_reward = self.mean_rewards[arm_index]
        mean_cost = self.mean_costs[arm_index]
        reward = self.rng.uniform() < mean_reward
        cost = self.rng.uniform() < mean_cost
        return reward, cost, mean_reward, mean_cost
