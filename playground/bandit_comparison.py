import multiprocessing
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

from components.bandit import AdaptiveBudgetedThompsonSampling
from components.bandits.thompson import ThompsonSampling
from components.bandits.abstract import AbstractBandit
from components.bandits.mrcb import MRCBBandit
from components.bandits.bts import BudgetedThompsonSampling
from components.bandits.ucbmb import UCBMBBandit
from components.bandits.ucb_variants import UCB
from components.bandits.kl_bucb import KLBUCB
from components.bandit_logging import BanditLogger
from util import run_async


def create_setting(k: int, high_variance: bool, seed: int):
    rng = np.random.default_rng(seed)
    scale = 0.8 if high_variance else 0.2
    low = (1 - scale) / 2
    high = 1 - low
    mean_rewards = rng.uniform(low, high, size=k)
    mean_costs = rng.uniform(low, high, size=k)
    return mean_rewards, mean_costs


def create_max_variance_setting(k: int, seed: int, low=0.02, high=0.98):
    rng = np.random.default_rng(seed)
    mean_rewards = rng.uniform(size=k, low=low, high=high)
    mean_costs = rng.uniform(size=k, low=low, high=high)
    return mean_rewards, mean_costs


def sort_setting(mean_rewards, mean_costs):
    ratio = mean_rewards / mean_costs
    sorted_indices = np.argsort(ratio)[::-1]
    return mean_rewards[sorted_indices], mean_costs[sorted_indices]


def create_bandits(k: int, seed: int):
    return np.array([
        # UCB(k=k, name="w-UCB", type="w", seed=seed),
        # KLBUCB(k=k, name="KLBUCB", seed=seed),
        # # AdaptiveBudgetedThompsonSampling(k=k, name="ABTS (hoeffding-t)", seed=seed,
        # #                                  ci_reward="hoeffding-t", ci_cost="hoeffding-t"),
        # # UCB(k=k, name="j-UCB", type="j", seed=seed),
        # UCB(k=k, name="i-UCB", type="i", seed=seed),
        # UCB(k=k, name="c-UCB", type="c", seed=seed),
        # UCB(k=k, name="m-UCB", type="m", seed=seed),
        # UCBMBBandit(k=k, name="UCB-MB", seed=seed),
        AdaptiveBudgetedThompsonSampling(k=k, name="ABTS (wilson-ci-t)", seed=seed,
                                         ci_reward="wilson-ci-t", ci_cost="wilson-ci-t"),
        # AdaptiveBudgetedThompsonSampling(k=k, name="ABTS (combined)", seed=seed,
        #                                  ci_reward="combined", ci_cost="combined"),
        AdaptiveBudgetedThompsonSampling(k=k, name="ABTS (wilson-ci)", seed=seed,
                                         ci_reward="wilson-ci", ci_cost="wilson-ci"),
        AdaptiveBudgetedThompsonSampling(k=k, name="ABTS (wilson)", seed=seed,
                                         ci_reward="wilson", ci_cost="wilson"),

        AdaptiveBudgetedThompsonSampling(k=k, name="ABTS (wilson-t)", seed=seed,
                                         ci_reward="wilson-t", ci_cost="wilson-t"),
        # ThompsonSampling(k=k, name="TS with costs", seed=seed),
        # ThompsonSampling(k=k, name="TS without costs", seed=seed),
        BudgetedThompsonSampling(k=k, name="BTS", seed=seed)
    ])


def iterate(bandit: AbstractBandit, mean_rewards, mean_costs, rng, logger):
    arm = bandit.sample()
    logger.track_arm(arm)
    mean_reward = mean_rewards[arm]
    mean_cost = mean_costs[arm]
    this_reward = int(rng.uniform() < mean_reward)
    this_cost = int(rng.uniform() < mean_cost)
    if isinstance(bandit, ThompsonSampling):
        if bandit.name == "TS with costs":
            normalized_reward = (
                                            1 + this_reward - this_cost) / 2  # gives 0 if mean_reward is much smaller than mean cost and 1 if mean cost is much smaller than mean reward
            this_normalized_reward = int(rng.uniform() < normalized_reward)
            bandit.update(arm, this_normalized_reward)
        else:
            bandit.update(arm, this_reward)
    else:
        bandit.update(arm, this_reward, this_cost)
    return this_reward, this_cost


def prepare_df(df: pd.DataFrame):
    df.ffill(inplace=True)
    df["total reward"] = df["reward"]
    df["spent budget"] = (df["spent-budget"] / 100).round() * 100
    df["regret"] = np.nan
    df["oracle"] = np.nan
    for _, gdf in df.groupby(["rep", "approach", "k", "high-variance"]):
        gdf["oracle"] = gdf["optimal-reward"] / gdf["optimal-cost"] * gdf["spent budget"]
        df["oracle"][gdf.index] = gdf["oracle"]
        df["regret"][gdf.index] = gdf["oracle"] - gdf["total reward"]
    df["k"] = df["k"].astype(int)
    return df


def plot_regret(df: pd.DataFrame):
    df = df[df["approach"] != "ABTS (wilson)"]
    # df = df[df["approach"] != "ABTS (hoeffding)"]
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


def run_bandit(bandit, B, mean_rewards, mean_costs, seed, hv):
    B_t = B
    logger = BanditLogger()
    logger.track_approach(bandit.name)
    logger.track_k(len(bandit))
    logger.track_high_variance(hv)
    logger.track_optimal_cost(mean_costs[0])
    logger.track_optimal_reward(mean_rewards[0])
    logger.track_rep(seed)
    rng = np.random.default_rng(seed)
    r_sum = 0
    while B_t > 0:
        r, c = iterate(bandit, mean_rewards, mean_costs, rng, logger)
        B_t -= c
        r_sum += r
        if (B_t % 10) == 1:
            logger.track_spent_budget(B - B_t)
            logger.track_reward(r_sum)
            logger.track_cost(c)
            logger.finalize_round()
    return logger.get_dataframe()


def get_best_arm_stats(df: pd.DataFrame):
    df["best arm identified"] = None
    for _, gdf in df.groupby(["approach", "rep", "k", "high-variance"]):
        best_arm = int(gdf["arm"].mode())
        gdf["best arm identified"] = best_arm
        df["best arm identified"][gdf.index] = best_arm == 0
    print(df["best arm identified"])
    sns.catplot(data=df, x="approach", y="best arm identified",
                col="high-variance", row="k",
                kind="count", ci=None)
    plt.show()


if __name__ == '__main__':
    use_results = False
    plot_results = False
    directory = os.path.join(os.getcwd(), "..", "results")
    filepath = os.path.join(directory, "bandit_comparison_ci.csv")
    assert os.path.exists(directory)
    if not use_results:
        high_variance = [True]
        ks = [10]
        B = 2000
        reps = 100
        dfs = []
        for k in tqdm(ks, desc="k"):
            for hv in tqdm(high_variance, leave=False, desc="variance"):
                all_args = []
                for rep in range(reps):
                    mean_rewards, mean_costs = create_setting(k, seed=rep, high_variance=True)
                    mean_rewards, mean_costs = sort_setting(mean_rewards, mean_costs)
                    args_list = [[b, B, mean_rewards, mean_costs, rep, hv] for b in create_bandits(k, rep)]
                    all_args.append(args_list)
                num_bandits = len(create_bandits(k, 0))
                for b_index in tqdm(range(num_bandits), leave=False, desc="Bandit"):
                    args_for_bandit = [a[b_index] for a in all_args]
                    results = run_async(run_bandit, args_for_bandit, njobs=multiprocessing.cpu_count() - 1)
                    dfs = dfs + results
        df = pd.concat(dfs)
        df.to_csv(filepath, index=False)
    if plot_results:
        df = pd.read_csv(filepath)
        df = prepare_df(df)
        # get_best_arm_stats(df)
        plot_regret(df)
