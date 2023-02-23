from typing import Tuple

import numpy as np

from components.experiments.abstract import Environment


class BernoulliSamplingEnvironment(Environment):
    def sample(self, arm_index: int) -> Tuple[int, int, float, float]:
        mean_reward = self.mean_rewards[arm_index]
        mean_cost = self.mean_costs[arm_index]
        reward = self.rng.uniform() < mean_reward
        cost = self.rng.uniform() < mean_cost
        return reward, cost, mean_reward, mean_cost


class BetaSamplingEnvironment(Environment):
    def __init__(self, mean_rewards: np.ndarray, mean_costs: np.ndarray, seed: int, min_param=1, max_param=100):
        super(BetaSamplingEnvironment, self).__init__(mean_rewards, mean_costs, seed)
        # first we sample some parameters
        self.alpha_r = self.rng.uniform(min_param, max_param, size=mean_rewards.shape)
        self.alpha_c = self.rng.uniform(min_param, max_param, size=mean_costs.shape)
        self.beta_r = self.rng.uniform(min_param, max_param, size=mean_rewards.shape)
        self.beta_c = self.rng.uniform(min_param, max_param, size=mean_costs.shape)
        # then we adjust the parameters to the mean_rewards and mean_costs
        rew_mask_less_05 = mean_rewards <= 0.5
        cost_mask_less_05 = mean_costs <= 0.5
        self.beta_r[rew_mask_less_05] = self.compute_beta_from_alpha(mean_rewards[rew_mask_less_05],
                                                                     self.alpha_r[rew_mask_less_05])
        self.alpha_r[rew_mask_less_05 == 0] = self.compute_alpha_from_beta(mean_rewards[rew_mask_less_05 == 0],
                                                                           self.beta_r[rew_mask_less_05 == 0])
        self.beta_c[cost_mask_less_05] = self.compute_beta_from_alpha(mean_costs[cost_mask_less_05],
                                                                      self.alpha_c[cost_mask_less_05])
        self.alpha_c[cost_mask_less_05 == 0] = self.compute_alpha_from_beta(mean_costs[cost_mask_less_05 == 0],
                                                                            self.beta_c[cost_mask_less_05 == 0])

    def compute_beta_from_alpha(self, mu, alpha):
        mu[mu == 0] = 0.0001
        return alpha * (1 / mu - 1)

    def compute_alpha_from_beta(self, mu, beta):
        mu[mu == 1] = 0.9999
        return beta * (mu / (1 - mu))

    def sample(self, arm_index: int) -> Tuple[float, float, float, float]:
        mean_reward = self.mean_rewards[arm_index]
        mean_cost = self.mean_costs[arm_index]
        reward = self.rng.beta(self.alpha_r[arm_index], self.beta_c[arm_index])
        cost = self.rng.beta(self.alpha_c[arm_index], self.beta_c[arm_index])
        return reward, cost, mean_reward, mean_cost
