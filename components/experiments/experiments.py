import multiprocessing
import os
from copy import deepcopy
from typing import List

from approach_names import *
from components.bandit_logging import *
from components.bandits.bts import BudgetedThompsonSampling
from components.bandits.ucb_variants import UCB
from components.bandits.ucbsc import UCBSC
from components.bandits.wucb import WUCB
from components.bandits.wucb_continuous import GeneralizedWUCB
from components.experiments.abstract import Experiment, Environment, execute_bandit_on_env
from components.experiments.environments import BernoulliSamplingEnvironment, BetaSamplingEnvironment
from facebook_ad_data_util import get_facebook_ad_data_settings, get_facebook_ad_stats
from util import run_async


class BernoulliExperiment(Experiment):
    def _generate_environments(self, k: int, seed: int) -> List[Environment]:
        rng = np.random.default_rng(seed)
        c_min = 0.01
        mean_rewards = rng.uniform(0, 1, size=k)
        mean_costs = rng.uniform(c_min, 1.0, size=k)
        eff_invers = mean_costs / mean_rewards
        sorted_indices = np.argsort(eff_invers)
        mean_rewards = mean_rewards[sorted_indices]
        mean_costs = mean_costs[sorted_indices]
        env = BernoulliSamplingEnvironment(mean_rewards=mean_rewards, mean_costs=mean_costs, seed=seed)
        return [env]


class BetaExperiment(Experiment):
    def _generate_environments(self, k: int, seed: int) -> List[Environment]:
        rng = np.random.default_rng(seed)
        c_min = 0.01
        mean_rewards = rng.uniform(0, 1, size=k)
        mean_costs = rng.uniform(c_min, 1.0, size=k)
        eff_invers = mean_costs / mean_rewards
        sorted_indices = np.argsort(eff_invers)
        mean_rewards = mean_rewards[sorted_indices]
        mean_costs = mean_costs[sorted_indices]
        env = BetaSamplingEnvironment(mean_rewards=mean_rewards, mean_costs=mean_costs, seed=seed)
        return [env]

    def _create_bandits(self, k: int, seed: int):
        return np.array([
            UCB(k=k, name=BUDGET_UCB, type="b", seed=seed),
            UCBSC(k=k, name=UCB_SC_PLUS, seed=seed),
            GeneralizedWUCB(k=k, name=ETA_UCB_1_6, seed=seed, r=1 / 6, adaptive=True),
            GeneralizedWUCB(k=k, name=ETA_UCB_1_5, seed=seed, r=1 / 5, adaptive=True),
            GeneralizedWUCB(k=k, name=ETA_UCB_1_4, seed=seed, r=1 / 4, adaptive=True),
            GeneralizedWUCB(k=k, name=ETA_UCB_1_3, seed=seed, r=1 / 3, adaptive=True),
            GeneralizedWUCB(k=k, name=ETA_UCB_1_2, seed=seed, r=1 / 2, adaptive=True),
            GeneralizedWUCB(k=k, name=ETA_UCB_1, seed=seed, r=1, adaptive=True),
            GeneralizedWUCB(k=k, name=ETA_UCB_2, seed=seed, r=2, adaptive=True),
            GeneralizedWUCB(k=k, name=ETA_UCB_3, seed=seed, r=3, adaptive=True),
            GeneralizedWUCB(k=k, name=ETA_UCB_4, seed=seed, r=4, adaptive=True),
            UCB(k=k, name=MUCB, type="m", seed=seed, adaptive=True),
            UCB(k=k, name=IUCB, type="i", seed=seed, adaptive=True),
            UCB(k=k, name=CUCB, type="c", seed=seed, adaptive=True),
            BudgetedThompsonSampling(k=k, name="BTS", seed=seed),
        ])


class FacebookBernoulliExperiment(Experiment):
    def _generate_environments(self, k: int,  # k is derived from the data in this environment
                               seed: int) -> List[Environment]:
        settings = get_facebook_ad_data_settings(random_state=seed)
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


class FacebookBetaExperiment(Experiment):
    def _generate_environments(self, k: int,  # k is derived from the data in this environment
                               seed: int) -> List[Environment]:
        settings = get_facebook_ad_data_settings(random_state=seed)
        envs = []
        for mean_rewards, mean_costs in settings:
            env = BetaSamplingEnvironment(mean_rewards=mean_rewards, mean_costs=mean_costs, seed=seed)
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

    def _create_bandits(self, k: int, seed: int):
        return np.array([
            UCB(k=k, name=BUDGET_UCB, type="b", seed=seed),
            UCBSC(k=k, name=UCB_SC_PLUS, seed=seed),
            GeneralizedWUCB(k=k, name=ETA_UCB_1_6, seed=seed, r=1 / 6, adaptive=True),
            GeneralizedWUCB(k=k, name=ETA_UCB_1_5, seed=seed, r=1 / 5, adaptive=True),
            GeneralizedWUCB(k=k, name=ETA_UCB_1_4, seed=seed, r=1 / 4, adaptive=True),
            GeneralizedWUCB(k=k, name=ETA_UCB_1_3, seed=seed, r=1 / 3, adaptive=True),
            GeneralizedWUCB(k=k, name=ETA_UCB_1_2, seed=seed, r=1 / 2, adaptive=True),
            GeneralizedWUCB(k=k, name=ETA_UCB_1, seed=seed, r=1, adaptive=True),
            GeneralizedWUCB(k=k, name=ETA_UCB_2, seed=seed, r=2, adaptive=True),
            GeneralizedWUCB(k=k, name=ETA_UCB_3, seed=seed, r=3, adaptive=True),
            GeneralizedWUCB(k=k, name=ETA_UCB_4, seed=seed, r=4, adaptive=True),
            UCB(k=k, name=MUCB, type="m", seed=seed, adaptive=True),
            UCB(k=k, name=IUCB, type="i", seed=seed, adaptive=True),
            UCB(k=k, name=CUCB, type="c", seed=seed, adaptive=True),
            BudgetedThompsonSampling(k=k, name="BTS", seed=seed),
        ])

    def run(self, arms: List[int], num_reps: int) -> pd.DataFrame:
        get_facebook_ad_stats()
        all_dfs = []
        all_args = self._generate_args(0,  # number of arms inferred from environments directly
                                       num_reps)
        dfs = run_async(execute_bandit_on_env, all_args, njobs=multiprocessing.cpu_count() - 1)
        all_dfs += dfs
        df = pd.concat(all_dfs)
        return df
