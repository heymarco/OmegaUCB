import multiprocessing
import os
import sys
from time import process_time_ns

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns

from util import run_async

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from components.bandit import AdaptiveBudgetedThompsonSampling
from components.bandits.bts import BudgetedThompsonSampling
from components.bandits.ucb_variants import UCB
from experiment import run_bandit, prepare_df


def prepare_data():
    data_path = os.path.join(os.getcwd(), "..", "data", "KAG_conversion_adapted.csv")
    raw_data = pd.read_csv(data_path)
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


def plot_regret(df: pd.DataFrame, filename: str):
    facet_kws = {'sharey': False, 'sharex': True}
    g = sns.relplot(data=df, kind="line",
                    x="normalized budget", y="regret",
                    hue="approach", row="p-min",
                    height=3, aspect=1, facet_kws=facet_kws,
                    ci=None)
    axes = g.axes.flatten()
    for ax in axes:
        ax.axhline(0, color="black", lw=.5)
    # plt.tight_layout(pad=.5)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", filename))
    plt.show()


def create_bandits(k: int, seed: int):
    return np.array([
        # UCB(k=k, name="j-UCB (a)", type="j", seed=seed, adaptive=True),
        # UCB(k=k, name="j-UCB", type="j", seed=seed, adaptive=False),
        UCB(k=k, name="w-UCB (a)", type="w", seed=seed, adaptive=True),
        UCB(k=k, name="w-UCB", type="w", seed=seed, adaptive=False),
        # UCB(k=k, name="i-UCB (a)", type="i", seed=seed, adaptive=True),
        # UCB(k=k, name="c-UCB", type="c", seed=seed),
        # UCB(k=k, name="m-UCB (a)", type="m", seed=seed, adaptive=True),
        # UCB(k=k, name="m-UCB", type="m", seed=seed, adaptive=False),
        AdaptiveBudgetedThompsonSampling(k=k, name="TS (cost)", seed=seed,
                                         ci_reward="ts-cost", ci_cost="ts-cost"),
        # AdaptiveBudgetedThompsonSampling(k=k, name="TS (reward)", seed=seed,
        #                                  ci_reward="ts-reward", ci_cost="ts-reward"),
        BudgetedThompsonSampling(k=k, name="BTS", seed=seed)
    ])


def plot_regret_over_k(df: pd.DataFrame):
    data = []
    for (k, approach, rep), gdf in df.groupby(["k", "approach", "rep"]):
        data.append([k, np.mean(gdf["regret"].iloc[-30:]), approach, rep])
    result_df = pd.DataFrame(data, columns=["Arms", "Regret", "Approach", "rep"])
    sns.barplot(data=result_df, x="Arms", y="Regret", hue="Approach")
    plt.yscale("log")
    plt.show()
    result_df = result_df[result_df["Arms"] > 3]
    result_df = result_df.groupby(["Approach"]).mean()
    result_df["Regret"] = result_df["Regret"] / np.min(result_df["Regret"])
    print(result_df)



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
        steps = 3 * 10e4  # we should be able to pull the cheapest arm 100000 times
        reps = 10
        dfs = []
        for (campaign_id, age, gender), gdf in tqdm(data.groupby(["campaign_id", "age", "gender"]), desc="Setting"):
            gdf = sort_df(gdf)
            mean_rewards, mean_costs = get_setting(gdf)
            all_args = []
            num_arms = len(mean_rewards)
            lowest_cost = np.min(mean_costs)
            print(lowest_cost)
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
        # plot_regret_over_k(df)
        plot_regret(df, filename + ".pdf")

