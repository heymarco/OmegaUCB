import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from components.bandit_logging import BanditLogger
from components.bandits.abstract import AbstractBandit


def iterate(bandit: AbstractBandit, mean_rewards, mean_costs, rng, logger):
    arm = bandit.sample()
    logger.track_arm(arm)
    mean_reward = mean_rewards[arm]
    mean_cost = mean_costs[arm]
    this_reward = int(rng.uniform() < mean_reward)
    this_cost = int(rng.uniform() < mean_cost)
    bandit.update(arm, this_reward, this_cost)
    return this_reward, this_cost


def prepare_df(df: pd.DataFrame, every_nth: int = 1):
    df.ffill(inplace=True)
    df = df.iloc[::every_nth]
    df["total reward"] = df["reward"]
    df["spent budget"] = np.round(df["spent-budget"] / 100) * 100
    df["regret"] = np.nan
    df["oracle"] = np.nan
    df["normalized budget"] = np.nan
    df["round"] = np.nan
    for group, gdf in df.groupby(["rep", "approach", "k", "high-variance", "p-min"]):
        gdf["oracle"] = gdf["optimal-reward"] / gdf["optimal-cost"] * gdf["spent-budget"]
        df["oracle"][gdf.index] = gdf["oracle"]
        normalized_budget = gdf["spent budget"] / gdf["spent budget"].iloc[-1]
        gdf["normalized budget"] = (normalized_budget * 3).round(1) / 3
        df["normalized budget"][gdf.index] = gdf["normalized budget"]
        gdf["regret"] = (gdf["oracle"] - gdf["total reward"]) / gdf["oracle"].iloc[-1]
        df["regret"][gdf.index] = gdf["regret"]
        df["round"][gdf.index] = np.arange(len(gdf))
    df["k"] = df["k"].astype(int)
    return df


def run_bandit(bandit, steps, B, mean_rewards, mean_costs, seed, hv, p_min):
    B_t = B
    logger = BanditLogger()
    logger.track_approach(bandit.name)
    logger.track_p_min(p_min)
    logger.track_k(len(bandit))
    logger.track_high_variance(hv)
    logger.track_optimal_cost(mean_costs[0])
    logger.track_optimal_reward(mean_rewards[0])
    logger.track_rep(seed)
    rng = np.random.default_rng(seed)
    r_sum = 0

    i = 0
    while B_t > 0:
        r, c = iterate(bandit, mean_rewards, mean_costs, rng, logger)
        B_t -= c
        r_sum += r
        if (i % 100) == 1:
            logger.track_iteration(B - B_t, r_sum, c)
            logger.finalize_round()
        i += 1
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