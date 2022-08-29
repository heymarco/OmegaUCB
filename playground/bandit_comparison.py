import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from components.bandit import ThompsonSampling, BudgetedThompsonSampling, AbstractBandit, AbstractArm, \
    AdaptiveBudgetedThompsonSampling
from components.bandit_logging import logger


def create_setting(k: int, high_variance: bool, seed: int):
    rng = np.random.default_rng(seed)
    scale = 0.8 if high_variance else 0.2
    low = (1 - scale) / 2
    high = 1 - low
    mean_rewards = rng.uniform(low, high, size=k)
    mean_costs = rng.uniform(low, high, size=k)
    return mean_rewards, mean_costs


def sort_setting(mean_rewards, mean_costs):
    ratio = mean_rewards / mean_costs
    sorted_indices = np.argsort(ratio)[::-1]
    return mean_rewards[sorted_indices], mean_costs[sorted_indices]


def create_bandits(k: int, seed: int):
    return np.array([AdaptiveBudgetedThompsonSampling(k=k, name="ABTS", seed=seed),
                     # ThompsonSampling(k=k, name="TS with costs", seed=seed),
                     # ThompsonSampling(k=k, name="TS without costs", seed=seed),
                     ThompsonSampling(k=1, name="Oracle", seed=seed),
                     BudgetedThompsonSampling(k=k, name="BTS", seed=seed)
                     ])


def run_bandit(bandit: AbstractBandit, mean_rewards, mean_costs):
    arm = bandit.sample()
    logger.track_arm(arm)
    mean_reward = mean_rewards[arm]
    mean_cost = mean_costs[arm]
    this_reward = int(np.random.uniform() < mean_reward)
    this_cost = int(np.random.uniform() < mean_cost)
    if isinstance(bandit, ThompsonSampling):
        if bandit.name == "TS with costs":
            normalized_reward = (1 + this_reward - this_cost) / 2  # gives 0 if mean_reward is much smaller than mean cost and 1 if mean cost is much smaller than mean reward
            this_normalized_reward = int(np.random.uniform() < normalized_reward)
            bandit.update(arm, this_normalized_reward)
        else:
            bandit.update(arm, this_reward)
    if isinstance(bandit, BudgetedThompsonSampling) or isinstance(bandit, AdaptiveBudgetedThompsonSampling):
        bandit.update(arm, this_reward, this_cost)
    return this_reward, this_cost


def iterate(bandits, mean_rewards, mean_costs):
    logger.track_optimal_reward(mean_rewards[0])
    logger.track_optimal_cost(mean_costs[0])
    for bandit in bandits:
        reward, cost = run_bandit(bandit, mean_rewards, mean_costs)
        logger.track_reward(reward)
        logger.track_cost(cost)
        logger.track_approach(bandit.name)
        logger.finalize_round()


def plot_regret(df: pd.DataFrame):
    df.ffill(inplace=True)
    df["round"] = np.nan
    df["regret"] = np.nan
    df["total reward"] = np.nan
    df["oracle"] = np.nan

    df["spent budget"] = (df["spent-budget"] / 5).round() * 5

    for _, gdf in df.groupby(["rep", "approach", "k", "high-variance"]):
        gdf["round"] = np.arange(1, len(gdf) + 1)
        df["round"][gdf.index] = gdf["round"]
        gdf["oracle"] = gdf["optimal-reward"] / gdf["optimal-cost"] * gdf["spent budget"]
        df["oracle"][gdf.index] = gdf["oracle"]
        gdf["total reward"] = gdf["reward"].cumsum()
        df["total reward"][gdf.index] = gdf["total reward"]
        df["regret"][gdf.index] = gdf["oracle"] - gdf["total reward"]

    df["k"] = df["k"].astype(int)
    facet_kws = {'sharey': False, 'sharex': True}
    g = sns.relplot(data=df, kind="line",
                x="spent budget", y="regret",
                hue="approach", row="k", col="high-variance",
                height=3, aspect=1, facet_kws=facet_kws,
                ci=None)
    axes = g.axes.flatten()
    for ax in axes:
        ax.axhline(0, color="black", lw=.5)
    # plt.tight_layout(pad=.5)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", "bandit_comparison.pdf"))
    plt.show()


if __name__ == '__main__':
    use_results = False
    plot_results = False
    directory = os.path.join(os.getcwd(), "..", "results")
    filepath = os.path.join(directory, "bandit_comparison.csv")
    assert os.path.exists(directory)
    if not use_results:
        high_variance = [True, False]
        ks = [100, 10, 4, 2]
        B = 1000
        reps = 300
        for k in tqdm(ks, desc="k"):
            logger.track_k(k)
            for hv in tqdm(high_variance, leave=False, desc="variance"):
                logger.track_high_variance(hv)
                for rep in tqdm(range(reps), leave=False, desc="rep"):
                    mean_rewards, mean_costs = create_setting(k=k, high_variance=hv, seed=rep)
                    mean_rewards, mean_costs = sort_setting(mean_rewards, mean_costs)
                    logger.track_optimal_cost(mean_costs[0])
                    logger.track_optimal_reward(mean_rewards[0])
                    logger.track_rep(rep)
                    for bandit in tqdm(create_bandits(k, rep), leave=False, desc="Bandits"):
                        B_t = B
                        logger.track_approach(bandit.name)
                        while B_t > 0:
                            r, c = run_bandit(bandit, mean_rewards, mean_costs)
                            B_t -= c
                            logger.track_spent_budget(B - B_t)
                            logger.track_reward(r)
                            logger.track_cost(c)
                            logger.finalize_round()
        df = logger.get_dataframe()
        df.to_csv(filepath, index=False)
    df = pd.read_csv(filepath)
    if plot_results:
        plot_regret(df)
