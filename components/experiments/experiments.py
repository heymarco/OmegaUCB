import multiprocessing
from copy import deepcopy
from typing import List

from approach_names import *
from components.bandit_logging import *
from components.bandits.bts import BudgetedThompsonSampling
from components.bandits.ucb_b2 import UCBB2
from components.bandits.ucb_variants import UCB
from components.bandits.ucbsc import UCBSC
from components.bandits.wucb import WUCB
from components.bandits.b_greedy import BGreedy
from components.bandits.w_star_ucb import WStarUCB
from components.experiments.abstract import Experiment, Environment, execute_bandit_on_env
from components.experiments.environments import BernoulliSamplingEnvironment, FacebookBetaSamplingEnvironment, RandomBetaSamplingEnvironment, MultinomialSamplingEnvironment
from facebook_ad_data_util import get_facebook_ad_data_settings
from util import run_async, create_multinomial_parameters, get_average_multinomial


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
        env = BernoulliSamplingEnvironment(mean_rewards=mean_rewards, mean_costs=mean_costs, rng=rng)
        return [env]

    def _create_bandits(self, k: int, seed: int):
        return np.array([
            UCBB2(k=k, name=UCB_B2_name, seed=seed),
            BGreedy(k=k, name="b-greedy", seed=seed),
            UCB(k=k, name=BUDGET_UCB, type="b", seed=seed),
            UCBSC(k=k, name=UCB_SC_PLUS, seed=seed),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_64, seed=seed, r=1 / 64, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_32, seed=seed, r=1 / 32, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_16, seed=seed, r=1 / 16, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_8, seed=seed, r=1 / 8, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_4, seed=seed, r=1 / 4, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_2, seed=seed, r=1 / 2, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1, seed=seed, r=1, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_2, seed=seed, r=2, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_64, seed=seed, r=1 / 64, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_32, seed=seed, r=1 / 32, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_16, seed=seed, r=1 / 16, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_8, seed=seed, r=1 / 8, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_4, seed=seed, r=1 / 4, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_2, seed=seed, r=1 / 2, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1, seed=seed, r=1, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_2, seed=seed, r=2, adaptive=True),
            UCB(k=k, name=MUCB, type="m", seed=seed, adaptive=True),
            UCB(k=k, name=IUCB, type="i", seed=seed, adaptive=True),
            UCB(k=k, name=CUCB, type="c", seed=seed, adaptive=True),
            BudgetedThompsonSampling(k=k, name="BTS", seed=seed),
        ])


class MultinomialExperiment(Experiment):
    def _generate_environments(self, k: int, seed: int) -> List[Environment]:
        rng = np.random.default_rng(seed)
        reward_params = create_multinomial_parameters(rng, k)
        cost_params = create_multinomial_parameters(rng, k)
        mean_rewards = get_average_multinomial(reward_params)
        mean_costs = get_average_multinomial(cost_params)
        eff_invers = mean_costs / mean_rewards
        sorted_indices = np.argsort(eff_invers)
        reward_params = reward_params[sorted_indices]
        cost_params = cost_params[sorted_indices]
        mean_rewards = mean_rewards[sorted_indices]
        mean_costs = mean_costs[sorted_indices]
        env = MultinomialSamplingEnvironment(mean_rewards=mean_rewards,
                                             mean_costs=mean_costs,
                                             rng=rng,
                                             reward_params=reward_params,
                                             cost_params=cost_params)
        return [env]

    def _create_bandits(self, k: int, seed: int):
        return np.array([
            UCBB2(k=k, name=UCB_B2_name, seed=seed),
            BGreedy(k=k, name="b-greedy", seed=seed),
            UCB(k=k, name=BUDGET_UCB, type="b", seed=seed),
            UCBSC(k=k, name=UCB_SC_PLUS, seed=seed),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_64, seed=seed, r=1 / 64, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_32, seed=seed, r=1 / 32, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_16, seed=seed, r=1 / 16, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_8, seed=seed, r=1 / 8, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_4, seed=seed, r=1 / 4, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_2, seed=seed, r=1 / 2, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1, seed=seed, r=1, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_2, seed=seed, r=2, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_64, seed=seed, r=1 / 64, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_32, seed=seed, r=1 / 32, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_16, seed=seed, r=1 / 16, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_8, seed=seed, r=1 / 8, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_4, seed=seed, r=1 / 4, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_2, seed=seed, r=1 / 2, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1, seed=seed, r=1, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_2, seed=seed, r=2, adaptive=True),
            UCB(k=k, name=MUCB, type="m", seed=seed, adaptive=True),
            UCB(k=k, name=IUCB, type="i", seed=seed, adaptive=True),
            UCB(k=k, name=CUCB, type="c", seed=seed, adaptive=True),
            BudgetedThompsonSampling(k=k, name="BTS", seed=seed),
        ])


class BetaExperiment(Experiment):
    def _generate_environments(self, k: int, seed: int) -> List[Environment]:
        rng = np.random.default_rng(seed)
        env = RandomBetaSamplingEnvironment(k=k, rng=rng)
        return [env]

    def _create_bandits(self, k: int, seed: int):
        return np.array([
            UCBB2(k=k, name=UCB_B2_name, seed=seed),
            BGreedy(k=k, name="b-greedy", seed=seed),
            UCB(k=k, name=BUDGET_UCB, type="b", seed=seed),
            UCBSC(k=k, name=UCB_SC_PLUS, seed=seed),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_64, seed=seed, r=1 / 64, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_32, seed=seed, r=1 / 32, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_16, seed=seed, r=1 / 16, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_8, seed=seed, r=1 / 8, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_4, seed=seed, r=1 / 4, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_2, seed=seed, r=1 / 2, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1, seed=seed, r=1, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_2, seed=seed, r=2, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_64, seed=seed, r=1 / 64, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_32, seed=seed, r=1 / 32, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_16, seed=seed, r=1 / 16, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_8, seed=seed, r=1 / 8, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_4, seed=seed, r=1 / 4, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_2, seed=seed, r=1 / 2, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1, seed=seed, r=1, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_2, seed=seed, r=2, adaptive=True),
            UCB(k=k, name=MUCB, type="m", seed=seed, adaptive=True),
            UCB(k=k, name=IUCB, type="i", seed=seed, adaptive=True),
            UCB(k=k, name=CUCB, type="c", seed=seed, adaptive=True),
            BudgetedThompsonSampling(k=k, name="BTS", seed=seed),
        ])


class FacebookBernoulliExperiment(Experiment):
    def _generate_environments(self, k: int,  # k is derived from the data in this environment
                               seed: int) -> List[Environment]:
        rng = np.random.default_rng(seed)
        settings = get_facebook_ad_data_settings()
        envs = []
        for mean_rewards, mean_costs in settings:
            env = BernoulliSamplingEnvironment(mean_rewards=mean_rewards, mean_costs=mean_costs, rng=rng)
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
        all_dfs = []
        all_args = self._generate_args(0,  # number of arms inferred from environments directly
                                       num_reps)
        dfs = run_async(execute_bandit_on_env, all_args, njobs=multiprocessing.cpu_count() - 1)
        all_dfs += dfs
        df = pd.concat(all_dfs)
        return df

    def _create_bandits(self, k: int, seed: int):
        return np.array([
            UCBB2(k=k, name=UCB_B2_name, seed=seed),
            BGreedy(k=k, name="b-greedy", seed=seed),
            UCB(k=k, name=BUDGET_UCB, type="b", seed=seed),
            UCBSC(k=k, name=UCB_SC_PLUS, seed=seed),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_64, seed=seed, r=1 / 64, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_32, seed=seed, r=1 / 32, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_16, seed=seed, r=1 / 16, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_8, seed=seed, r=1 / 8, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_4, seed=seed, r=1 / 4, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_2, seed=seed, r=1 / 2, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1, seed=seed, r=1, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_2, seed=seed, r=2, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_64, seed=seed, r=1 / 64, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_32, seed=seed, r=1 / 32, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_16, seed=seed, r=1 / 16, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_8, seed=seed, r=1 / 8, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_4, seed=seed, r=1 / 4, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_2, seed=seed, r=1 / 2, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1, seed=seed, r=1, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_2, seed=seed, r=2, adaptive=True),
            UCB(k=k, name=MUCB, type="m", seed=seed, adaptive=True),
            UCB(k=k, name=IUCB, type="i", seed=seed, adaptive=True),
            UCB(k=k, name=CUCB, type="c", seed=seed, adaptive=True),
            BudgetedThompsonSampling(k=k, name="BTS", seed=seed),
        ])


class FacebookBetaExperiment(Experiment):
    def _generate_environments(self, k: int,  # k is derived from the data in this environment
                               seed: int) -> List[Environment]:
        rng = np.random.default_rng(seed)
        settings = get_facebook_ad_data_settings()
        envs = [
            FacebookBetaSamplingEnvironment(mean_rewards=mean_rewards, mean_costs=mean_costs, rng=rng)
            for mean_rewards, mean_costs in settings
        ]
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
            UCBB2(k=k, name=UCB_B2_name, seed=seed),
            BGreedy(k=k, name="b-greedy", seed=seed),
            UCB(k=k, name=BUDGET_UCB, type="b", seed=seed),
            UCBSC(k=k, name=UCB_SC_PLUS, seed=seed),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_64, seed=seed, r=1 / 64, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_32, seed=seed, r=1 / 32, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_16, seed=seed, r=1 / 16, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_8, seed=seed, r=1 / 8, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_4, seed=seed, r=1 / 4, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1_2, seed=seed, r=1 / 2, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_1, seed=seed, r=1, adaptive=True),
            WStarUCB(k=k, name=OMEGA_STAR_UCB_2, seed=seed, r=2, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_64, seed=seed, r=1 / 64, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_32, seed=seed, r=1 / 32, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_16, seed=seed, r=1 / 16, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_8, seed=seed, r=1 / 8, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_4, seed=seed, r=1 / 4, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1_2, seed=seed, r=1 / 2, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_1, seed=seed, r=1, adaptive=True),
            WUCB(k=k, name=OMEGA_UCB_2, seed=seed, r=2, adaptive=True),
            UCB(k=k, name=MUCB, type="m", seed=seed, adaptive=True),
            UCB(k=k, name=IUCB, type="i", seed=seed, adaptive=True),
            UCB(k=k, name=CUCB, type="c", seed=seed, adaptive=True),
            BudgetedThompsonSampling(k=k, name="BTS", seed=seed),
        ])

    def run(self, arms: List[int], num_reps: int) -> pd.DataFrame:
        all_dfs = []
        all_args = self._generate_args(0,  # number of arms inferred from environments directly
                                       num_reps)
        dfs = run_async(execute_bandit_on_env, all_args, njobs=multiprocessing.cpu_count() - 1)
        all_dfs += dfs
        df = pd.concat(all_dfs)
        return df
