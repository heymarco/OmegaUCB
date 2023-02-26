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
    def __init__(self, ar, br, ac, bc, mean_rewards, mean_costs, rng):
        super(BetaSamplingEnvironment, self).__init__(mean_rewards, mean_costs, rng)
        self.alpha_r = ar
        self.beta_r = br
        self.alpha_c = ac
        self.beta_c = bc

    def sample(self, arm_index: int) -> Tuple[float, float, float, float]:
        mean_reward = self.mean_rewards[arm_index]
        mean_cost = self.mean_costs[arm_index]
        reward = self.rng.beta(self.alpha_r[arm_index],
                               self.beta_r[arm_index])
        cost = self.rng.beta(self.alpha_c[arm_index],
                             self.beta_c[arm_index])
        return reward, cost, mean_reward, mean_cost

    def _get_sorted_indices(self, ar, br, ac, bc):
        mean_r = ar / (ar + br)
        mean_c = ac / (ac + bc)
        eff_inverse = mean_c / mean_r
        return np.argsort(eff_inverse)


class RandomBetaSamplingEnvironment(BetaSamplingEnvironment):
    def __init__(self, k: int, rng, min_param=0, max_param=5):
        # first we sample some parameters
        alpha_r = rng.uniform(min_param, max_param, size=k)
        alpha_c = rng.uniform(min_param, max_param, size=k)
        beta_r = rng.uniform(min_param, max_param, size=k)
        beta_c = rng.uniform(min_param, max_param, size=k)
        sorted_indices = self._get_sorted_indices(ar=alpha_r, ac=alpha_c, br=beta_r, bc=beta_c)
        alpha_r = alpha_r[sorted_indices]
        beta_r = beta_r[sorted_indices]
        alpha_c = alpha_c[sorted_indices]
        beta_c = beta_c[sorted_indices]
        mean_rewards = alpha_r / (alpha_r + beta_r)
        mean_costs = alpha_c / (alpha_c + beta_c)
        super(RandomBetaSamplingEnvironment, self).__init__(ar=alpha_r, br=beta_r, ac=alpha_c, bc=beta_c,
                                                            mean_rewards=mean_rewards, mean_costs=mean_costs, rng=rng)


class FacebookBetaSamplingEnvironment(BetaSamplingEnvironment):
    def __init__(self, mean_rewards: np.ndarray, mean_costs: np.ndarray, rng, min_param=0, max_param=5):
        # first we sample some parameters
        alpha_r = rng.uniform(min_param, max_param, size=mean_rewards.shape)
        alpha_c = rng.uniform(min_param, max_param, size=mean_rewards.shape)
        beta_r = rng.uniform(min_param, max_param, size=mean_costs.shape)
        beta_c = rng.uniform(min_param, max_param, size=mean_costs.shape)
        rew_mask_less_05 = mean_rewards <= 0.5
        cost_mask_less_05 = mean_costs <= 0.5
        beta_r[rew_mask_less_05] = self.compute_beta_from_alpha(mean_rewards[rew_mask_less_05],
                                                                alpha_r[rew_mask_less_05])
        alpha_r[rew_mask_less_05 == 0] = self.compute_alpha_from_beta(mean_rewards[rew_mask_less_05 == 0],
                                                                      beta_r[rew_mask_less_05 == 0])
        beta_c[cost_mask_less_05] = self.compute_beta_from_alpha(mean_costs[cost_mask_less_05],
                                                                 alpha_c[cost_mask_less_05])
        alpha_c[cost_mask_less_05 == 0] = self.compute_alpha_from_beta(mean_costs[cost_mask_less_05 == 0],
                                                                       beta_c[cost_mask_less_05 == 0])

        super(FacebookBetaSamplingEnvironment, self).__init__(ar=alpha_r, br=beta_r, ac=alpha_c, bc=beta_c,
                                                              mean_rewards=mean_rewards, mean_costs=mean_costs, rng=rng)

    def compute_beta_from_alpha(self, mu, alpha):
        mu[mu == 0] = 0.0001
        return alpha * (1 / mu - 1)

    def compute_alpha_from_beta(self, mu, beta):
        mu[mu == 1] = 0.9999
        return beta * (mu / (1 - mu))