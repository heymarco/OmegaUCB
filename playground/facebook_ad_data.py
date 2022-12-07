import multiprocessing
import os
import sys
from time import process_time_ns

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{times}'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{nicefrac}'
mpl.rc('font', family='serif')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from util import run_async, load_facebook_data
from components.bandit import AdaptiveBudgetedThompsonSampling
from components.bandits.bts import BudgetedThompsonSampling
from components.bandits.ucb_variants import UCB


def prepare_data():
    raw_data = load_facebook_data()
    is_zero_cost = raw_data["spent"] == 0
    is_zero_reward = raw_data["approved_conversion"] == 0
    is_nan_ratio = np.isnan(raw_data["reward_cost_ratio"])
    non_informative_rows = np.logical_and(is_zero_reward, is_zero_cost)  # do not include ads for which we have no data
    corrupted_rows = np.logical_and(np.invert(is_zero_reward), is_zero_cost)  # cost although no clicks occurred
    mask = np.invert(np.logical_or(non_informative_rows, corrupted_rows))
    mask = np.logical_or(mask, np.invert(is_nan_ratio))
    filtered_df = raw_data.loc[mask].reset_index()
    high_ratio_outliers = filtered_df["reward_cost_ratio"] > 1
    filtered_df = filtered_df.loc[np.invert(high_ratio_outliers)]
    high_revenue_outliers = filtered_df["revenue_per_1000_impressions"] > 1
    filtered_df = filtered_df.loc[np.invert(high_revenue_outliers)]
    return filtered_df


def sort_df(df):
    df = df.sort_values(by="reward_cost_ratio", ascending=False)
    return df


def get_setting(df):
    mean_rewards = list(df["revenue_per_1000_impressions"])
    mean_costs = list(df["cost_per_1000_impressions"])
    return mean_rewards, mean_costs


def create_bandits(k: int, seed: int):
    return np.array([
        # UCB(k=k, name="j-UCB (a)", type="j", seed=seed, adaptive=True),
        # UCB(k=k, name="j-UCB", type="j", seed=seed, adaptive=False),
        UCB(k=k, name="w-UCB (a)", type="w", seed=seed, adaptive=True),
        UCB(k=k, name="w-UCB", type="w", seed=seed, adaptive=False),
        # UCB(k=k, name="i-UCB (a)", type="i", seed=seed, adaptive=True),
        # UCB(k=k, name="c-UCB", type="c", seed=seed),
        UCB(k=k, name="m-UCB (a)", type="m", seed=seed, adaptive=True),
        # UCB(k=k, name="m-UCB", type="m", seed=seed, adaptive=False),
        AdaptiveBudgetedThompsonSampling(k=k, name="TS (cost)", seed=seed,
                                         ci_reward="ts-cost", ci_cost="ts-cost"),
        # AdaptiveBudgetedThompsonSampling(k=k, name="TS (reward)", seed=seed,
        #                                  ci_reward="ts-reward", ci_cost="ts-reward"),
        BudgetedThompsonSampling(k=k, name="BTS", seed=seed)
    ])


def plot_regret(df: pd.DataFrame, filename: str):
    facet_kws = {'sharey': False, 'sharex': True}
    g = sns.relplot(data=df, kind="line",
                    x="normalized budget", y="regret", row="p-min",
                    hue="approach",
                    height=3, aspect=1, facet_kws=facet_kws,
                    ci=None)
    axes = g.axes.flatten()
    for ax in axes:
        ax.axhline(0, color="black", lw=.5)
    # plt.tight_layout(pad=.5)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", filename))
    plt.show()


def plot_regret_over_k(df: pd.DataFrame):
    data = []
    for (k, p_min, approach, rep), gdf in df.groupby(["k", "p-min", "approach", "rep"]):
        data.append([k, p_min, np.mean(gdf["regret"].iloc[-30:]), approach, rep])
    result_df = pd.DataFrame(data, columns=["k", "c-min", "Regret", "Approach", "rep"])
    result_df = result_df[result_df["k"] > 3]
    sns.lineplot(data=result_df, x="c-min", y="Regret", hue="Approach", marker="o")
    plt.yscale("log")
    plt.show()



if __name__ == '__main__':
    use_results = True
    plot_results = True
    directory = os.path.join(os.getcwd(), "..", "results")
    filename = "bandit_comparison_facebook_ads"
    filepath = os.path.join(directory, filename + ".csv")
    assert os.path.exists(directory)

    data = prepare_data()

    if not use_results:
        high_variance = [True]
        steps = 1e5  # we should be able to pull the cheapest arm 10000 times
        reps = 200
        dfs = []
        for (campaign_id, age, gender), gdf in tqdm(data.groupby(["campaign_id", "age", "gender"]), desc="Setting"):
            gdf = sort_df(gdf)
            mean_rewards, mean_costs = get_setting(gdf)
            all_args = []
            num_arms = len(mean_rewards)
            lowest_cost = np.min(mean_costs)
            B = int(np.ceil(steps * lowest_cost))
            if num_arms <= 1:
                continue
            for rep in range(reps):
                args_list = [[b, int(steps), B, mean_rewards, mean_costs, rep, False,
                              str(campaign_id) + "-" + str(age) + "-" + str(gender)]
                             for b in create_bandits(num_arms, rep)]
                all_args.append(args_list)
            for b_index in tqdm(range(len(create_bandits(num_arms, 0))), leave=False, desc="Bandit"):
                args_for_bandit = [a[b_index] for a in all_args]
                results = run_async(run_bandit, args_for_bandit, njobs=multiprocessing.cpu_count() - 1)
                dfs = dfs + results
        df = pd.concat(dfs)
        df.to_csv(filepath, index=False)
    if plot_results:
        df = pd.read_csv(filepath)
        df = prepare_df(df)
        for index, ((campaign_id, age, gender), gdf) in enumerate(data.groupby(["campaign_id", "age", "gender"])):
            gdf = sort_df(gdf)
            mean_rewards, mean_costs = get_setting(gdf)
            print("Rew", mean_rewards)
            print("Cos", mean_costs)
            p_min = str(campaign_id) + "-" + str(age) + "-" + str(gender)
            df["p-min"].iloc[df["p-min"] == p_min] = mean_costs[0]
        print(df["p-min"])
        plot_regret(df, filename + ".pdf")
        plot_regret_over_k(df)

