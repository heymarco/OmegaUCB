import os
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from components.bandit_logging import *
from components.bandits.abstract import AbstractBandit
from components.experiments.abstract import Experiment, Environment
from components.experiments.environments import BernoulliSamplingEnvironment


def prepare_df2(df: pd.DataFrame):
    df.ffill(inplace=True)
    df[NORMALIZED_BUDGET] = df[NORMALIZED_BUDGET].round(1)
    return df


def prepare_df(df: pd.DataFrame, every_nth: int = 1):
    df.ffill(inplace=True)
    df = df.iloc[::every_nth]
    df["total reward"] = df["reward"]
    # df["spent budget"] = np.round(df["spent-budget"] / 100) * 100
    df["regret"] = np.nan
    df["oracle"] = np.nan
    df["normalized budget"] = np.nan
    df["round"] = np.nan
    df["our_approach"] = df["approach"].apply(lambda x: "w-UCB" in x)
    df["diff"] = np.nan

    for group, gdf in df.groupby(["rep", "approach", "k", "high-variance", "p-min"]):
        gdf["diff"] = np.diff(gdf["spent-budget"], prepend=0.0)
        df["diff"][gdf.index] = gdf["diff"]
    df = df.loc[df["diff"] > 0]

    for group, gdf in df.groupby(["rep", "approach", "k", "high-variance", "p-min"]):
        rounds = (gdf.index.to_numpy() - gdf.index.to_numpy()[0]) * 100 + 1 + every_nth
        gdf["round"] = rounds
        gdf["oracle"] = gdf["optimal-reward"] * gdf["spent-budget"] / gdf["optimal-cost"]
        df["oracle"][gdf.index] = gdf["oracle"]
        normalized_budget = gdf["spent-budget"] / gdf["spent-budget"].iloc[-1]
        gdf["regret"] = gdf["oracle"] - gdf["total reward"]
        df["regret"][gdf.index] = gdf["regret"]
        gdf["normalized budget"] = np.ceil((normalized_budget * 100)) / 100
        df["normalized budget"][gdf.index] = gdf["normalized budget"]
    budget_points = np.arange(1, 21) / 20
    df_mask = df["normalized budget"].apply(lambda x: x in budget_points)
    df = df.loc[df_mask]
    df["k"] = df["k"].astype(int)
    return df


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