import multiprocessing
import os
from copy import deepcopy
from typing import List

from components.bandit_logging import *
from components.experiments.abstract import Experiment, Environment, execute_bandit_on_env
from components.experiments.environments import BernoulliSamplingEnvironment
from facebook_ad_data_util import get_facebook_ad_data_settings, get_facebook_ad_stats
from util import run_async


class UniformArmsExperiment(Experiment):
    def _generate_environments(self, k: int, seed: int) -> List[Environment]:
        rng = np.random.default_rng(seed)
        c_mins = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        envs = []
        for c in c_mins:
            low = c
            high = 1.0
            mean_rewards = rng.uniform(low, high, size=k - 1)
            max_reward = np.max(mean_rewards)
            mean_costs = rng.uniform(low, high, size=k - 1)
            mean_rewards = np.concatenate([[max_reward], mean_rewards])
            mean_costs = np.concatenate([[c], mean_costs])  # we want to include the minimum cost
            env = BernoulliSamplingEnvironment(mean_rewards=mean_rewards, mean_costs=mean_costs, seed=seed)
            envs.append(env)
        return envs


class FacebookAdDataExperiment(Experiment):
    def _generate_environments(self, k: int,  # k is derived from the data in this environment
                               seed: int) -> List[Environment]:
        settings = get_facebook_ad_data_settings()
        envs = []
        for mean_rewards, mean_costs in settings:
            env = BernoulliSamplingEnvironment(mean_rewards=mean_rewards, mean_costs=mean_costs, seed=seed)
            envs.append(env)
        return envs

    def _generate_args(self, k: int, num_reps: int) -> List:
        args = []
        for rep in range(num_reps):
            environments = self._generate_environments(k=k, seed=rep)
            for env in environments:
                k = len(env.mean_rewards)
                bandits = self._create_bandits(k=k, seed=rep)
                for bandit in bandits:
                    args.append([deepcopy(bandit), deepcopy(env), self.num_steps, rep])
        return args

    def run(self, arms: List[int], num_reps: int) -> pd.DataFrame:
        get_facebook_ad_stats()
        all_dfs = []
        all_args = self._generate_args(0,  # number of arms inferred from environments directly
                                       num_reps)
        dfs = run_async(execute_bandit_on_env, all_args, njobs=multiprocessing.cpu_count() - 1)
        all_dfs += dfs
        df = pd.concat(all_dfs)
        return df


