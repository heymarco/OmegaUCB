import multiprocessing

import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import sys
import os
from time import process_time_ns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from components.bandit import AdaptiveBudgetedThompsonSampling
from components.bandits.bts import BudgetedThompsonSampling
from components.bandits.ucb_variants import UCB
from util import run_async, subsample_csv
from experiment import prepare_df, run_bandit


def create_setting(k: int, high_variance: bool, p_min, seed: int):
    rng = np.random.default_rng(seed)
    low = p_min
    high = 1.0 if high_variance else min(1.0, p_min * 3)
    mean_rewards = np.array([0.5 for _ in range(k)])  # investigate effect when all arms have same cost.
    mean_costs = rng.uniform(low, high, size=k)
    return mean_rewards, mean_costs


def create_max_variance_setting(k: int, seed: int, low=0.1, high=0.9):
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
    facet_kws = {'sharey': False, 'sharex': False}
    df["normalized budget"] = df.index
    g = sns.relplot(data=df, kind="line",
                    x="normalized budget", y="regret",
                    hue="approach", row="k", col="p-min",
                    height=3, aspect=1, facet_kws=facet_kws,
                    ci=None)
    axes = g.axes.flatten()
    for ax in axes:
        ax.axhline(0, color="black", lw=.5)
    # plt.tight_layout(pad=.5)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", filename))
    plt.show()


if __name__ == '__main__':
    use_results = True
    plot_results = True
    directory = os.path.join(os.getcwd(), "..", "results")
    filename = "bandit_comparison_ci"
    filepath = os.path.join(directory, filename + ".csv")
    assert os.path.exists(directory)
    if not use_results:
        high_variance = [True]
        p_min = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5]
        ks = [10]
        steps = 5*10e3  # we should be able to pull the cheapest arm 100000 times
        reps = 10
        dfs = []
        for k in tqdm(ks, desc="k"):
            for hv in tqdm(high_variance, leave=False, desc="variance"):
                for p in tqdm(p_min, leave=False, desc="p_min"):
                    all_args = []
                    process_start = process_time_ns()
                    for rep in range(reps):
                        mean_rewards, mean_costs = create_setting(k, seed=rep, high_variance=hv, p_min=p)
                        mean_rewards, mean_costs = sort_setting(mean_rewards, mean_costs)
                        lowest_cost = np.min(mean_costs)
                        B = int(np.ceil(steps * lowest_cost))
                        args_list = [[b, int(steps), B, mean_rewards, mean_costs, rep, hv, p]
                                     for b in create_bandits(k, rep)]
                        all_args.append(args_list)
                    num_bandits = len(create_bandits(k, 0))
                    for b_index in tqdm(range(num_bandits), leave=False, desc="Bandit"):
                        args_for_bandit = [a[b_index] for a in all_args]
                        results = run_async(run_bandit, args_for_bandit, njobs=multiprocessing.cpu_count() - 1)
                        dfs = dfs + results
        df = pd.concat(dfs)
        df.to_csv(filepath, index=False)
    if plot_results:
        subsample_csv(filepath, every_nth=500)
        reduced_filename = filename + "_reduced" 
        reduced_filepath = os.path.join(directory, reduced_filename)
        df = pd.read_csv(reduced_filepath + ".csv")
        df = prepare_df(df)
        plot_regret(df, reduced_filename + ".pdf")
