import multiprocessing
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Tuple, List

from tqdm import tqdm

from components.bandit_logging import *
from components.bandits.abstract import AbstractBandit
from components.bandits.ucb_b2 import UCBB2
from components.bandits.ucb_variants import UCB
from util import run_async, incremental_regret


class Environment(ABC):
    def __init__(self, mean_rewards: np.ndarray, mean_costs: np.ndarray, rng):
        self.mean_rewards = mean_rewards
        self.mean_costs = mean_costs
        self.rng = rng

    @abstractmethod
    def sample(self, arm_index: int) -> Tuple[float, float, float, float]:
        """
        Observe reward and cost from this environment by pulling arm with index `arm_index`
        :param arm_index: the index of the pulled arm
        :return: reward, cost, mean reward, mean cost
        """
        raise NotImplementedError

    def get_stats(self):
        """
        Get stats about the environment
        :return: dict with best arm index and minimum average cost (not necessarily the same as the cost of the best arm)
        """
        efficiency = self.mean_rewards / self.mean_costs
        best_arm_index = np.argmax(efficiency)
        c_min = np.min(self.mean_costs)
        return {
            BEST_ARM: best_arm_index,
            MINIMUM_AVERAGE_COST: c_min
        }


class Experiment(ABC):
    def __init__(self, name: str, num_steps=1e5):
        self.name = name
        self.num_steps = num_steps

    @abstractmethod
    def _generate_environments(self, k: int, seed: int) -> List[Environment]:
        raise NotImplementedError

    def _create_bandits(self, k: int, seed: int):
        raise NotImplementedError

    def _generate_args(self, k: int, num_reps: int) -> List:
        args = []
        for rep in range(num_reps):
            environments = self._generate_environments(k=k, seed=rep)
            bandits = self._create_bandits(k=k, seed=rep)
            for env in environments:
                for bandit in bandits:
                    args.append([deepcopy(bandit), deepcopy(env), self.num_steps, rep])
        return args

    def run(self, arms: List[int], num_reps: int) -> pd.DataFrame:
        all_dfs = []
        for k in tqdm(arms):
            all_args = self._generate_args(k, num_reps)
            dfs = run_async(execute_bandit_on_env, all_args, njobs=multiprocessing.cpu_count())
            all_dfs += dfs
        df = pd.concat(all_dfs)
        return df


def iterate(bandit: AbstractBandit, env: Environment, rng):
    if isinstance(bandit, UCB) or isinstance(bandit, UCBB2):
        c_min = env.get_stats()[MINIMUM_AVERAGE_COST]
        arm = bandit.sample(c_min=c_min)
    else:
        arm = bandit.sample()
    reward, cost, mean_reward, mean_cost = env.sample(arm)
    bandit.update(arm, reward, cost)
    return reward, cost, arm


def execute_bandit_on_env(bandit: AbstractBandit, env: Environment, num_steps: int, rep: int) -> pd.DataFrame:
    logger = BanditLogger()
    spent_budget = 0
    expected_spent_budget = 0
    env_stats = env.get_stats()
    logger.track_approach(bandit.name)
    logger.track_c_min(env_stats[MINIMUM_AVERAGE_COST])
    logger.track_k(len(bandit))
    cost_optimal_arm = env.mean_costs[0]
    reward_optimal_arm = env.mean_rewards[0]
    logger.track_optimal_cost(cost_optimal_arm)
    logger.track_optimal_reward(reward_optimal_arm)
    logger.track_rep(rep)
    rng = np.random.default_rng(rep)
    r_sum = 0
    regret_sum = 0
    budget = num_steps * env_stats[MINIMUM_AVERAGE_COST]

    t = 0
    while spent_budget < budget:
        r, c, arm = iterate(bandit, env, rng)
        spent_budget += c
        r_sum += r
        mu_r_this = env.mean_rewards[arm]
        mu_c_this = env.mean_costs[arm]
        expected_spent_budget += mu_c_this
        regret_sum += incremental_regret(rew_this=mu_r_this, cost_this=mu_c_this,
                                         rew_best=reward_optimal_arm, cost_best=cost_optimal_arm)
        should_track = t % int(1e3) == 0
        if should_track:
            logger.track_arm(arm)
            logger.track_round(t)
            logger.track_regret(regret_sum)
            logger.track_mean_rew_current_arm(mu_r_this)
            logger.track_mean_cost_current_arm(mu_c_this)
            logger.track_spent_budget(spent_budget)
            logger.track_expected_spent_budget(expected_spent_budget)
            logger.track_normalized_budget(spent_budget / budget)
            logger.track_total_reward(r_sum)
            logger.track_time()
            logger.finalize_round()
        t += 1
    return logger.get_dataframe()




