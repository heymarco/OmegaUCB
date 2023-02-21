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


from components.bandits.bts import BudgetedThompsonSampling
from components.bandits.ucb_variants import UCB
from components.bandits.ucbsc import UCBSC
from components.bandits.wucb import WUCB
from util import run_async, create_palette, cm2inch, extract_rho
from experiment import prepare_df, run_bandit


import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = rho'\usepackage{times}'
mpl.rcParams['text.latex.preamble'] = rho'\usepackage{nicefrac}'
mpl.rc('font', family='serif')


def create_setting(k: int, high_variance: bool, c_min, seed: int):
    rng = np.random.default_rng(seed)
    low = c_min
    high = 1.0 if high_variance else min(1.0, c_min * 3)
    mean_rewards = rng.uniform(low, high, size=k-1)
    max_reward = np.max(mean_rewards)
    mean_costs = rng.uniform(low, high, size=k-1)
    mean_rewards = np.concatenate([[max_reward], mean_rewards])
    mean_costs = np.concatenate([[c_min], mean_costs])  # we want to include the minimum cost
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
        UCBSC(k=k, name="UCB-SC+", seed=seed),
        WUCB(k=k, name="w-UCB (a, rho=1/6)", seed=seed, r=1/6, adaptive=True),
        WUCB(k=k, name="w-UCB (a, rho=1/5)", seed=seed, r=1/5, adaptive=True),
        WUCB(k=k, name="w-UCB (a, rho=1/4)", seed=seed, r=1/4, adaptive=True),
        WUCB(k=k, name="w-UCB (a, rho=1/3)", seed=seed, r=1/3, adaptive=True),
        WUCB(k=k, name="w-UCB (a, rho=1/2)", seed=seed, r=1/2, adaptive=True),
        WUCB(k=k, name="w-UCB (a, rho=1)", seed=seed, r=1, adaptive=True),
        WUCB(k=k, name="w-UCB (a, rho=2)", seed=seed, r=2, adaptive=True),
        WUCB(k=k, name="w-UCB (a, rho=3)", seed=seed, r=3, adaptive=True),
        WUCB(k=k, name="w-UCB (a, rho=4)", seed=seed, r=4, adaptive=True),
        UCB(k=k, name="m-UCB", type="m", seed=seed, adaptive=True),
        UCB(k=k, name="i-UCB", type="i", seed=seed, adaptive=True),
        UCB(k=k, name="c-UCB", type="c", seed=seed, adaptive=True),
        BudgetedThompsonSampling(k=k, name="BTS", seed=seed),
    ])


def plot_regret(df: pd.DataFrame, filename: str):
    df.sort_values(by="approach", inplace=True)
    facet_kws = {'sharey': False, 'sharex': True}
    palette = create_palette(df)
    g = sns.relplot(data=df, kind="line",
                    x="normalized budget", y="regret",
                    row="p-min", col="k",
                    hue="approach",
                    palette=palette,
                    height=cm2inch(4)[0], aspect=1.5, facet_kws=facet_kws,
                    ci=None)
    axes = g.axes.flatten()
    for ax in axes:
        ax.axhline(0, color="black", lw=.5)
    # plt.tight_layout(pad=.5)
    # g.set(xscale="log")
    # g.set(yscale="log")
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", filename))
    plt.show()


def plot_nrounds(df: pd.DataFrame, filename: str):
    df.sort_values(by="approach", inplace=True)
    facet_kws = {'sharey': False, 'sharex': True}
    palette = create_palette(df)
    g = sns.relplot(data=df, kind="line",
                    x="normalized budget", y="round",
                    row="p-min", col="k",
                    hue="approach", palette=palette,
                    height=cm2inch(6), aspect=1, facet_kws=facet_kws,
                    ci=None)
    axes = g.axes.flatten()
    for ax in axes:
        ax.axhline(0, color="black", lw=.5)
    # plt.tight_layout(pad=.5)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", filename))
    plt.show()

def plot_regret_over_k(df: pd.DataFrame):
    data = []
    for (k, approach, rep), gdf in df.groupby(["k", "approach", "rep"]):
        data.append([k, np.mean(gdf["regret"].iloc[-30:]), approach, rep])
    result_df = pd.DataFrame(data, columns=["Arms", "Regret", "Approach", "rep"])
    result_df = result_df[result_df["Arms"] > 3]
    df["p-min"] = df["p-min"].astype(float)
    sns.lineplot(data=result_df, x="p-min", y="Regret", hue="Approach", marker="o")
    plt.show()


def plot_regret_over_k(df: pd.DataFrame):
    df.sort_values(by="approach", inplace=True)
    data = []
    palette = create_palette(df)
    for (approach, k, p_min, rep), gdf in df.groupby(["approach", "k", "p-min", "rep"]):
        data.append([k, p_min, np.mean(gdf["regret"].iloc[-30:]), approach, rep])
    result_df = pd.DataFrame(data, columns=["k", rho"$c_{min}$", "Regret", "Approach", "rep"])
    result_df = result_df[result_df["k"] > 3]
    # g = sns.lineplot(data=result_df, x=rho"$c_{min}$", y="Regret", hue="Approach", marker="o",
    #                  palette=palette,
    #                  err_style="bars")
    sns.boxplot(data=result_df, x=rho"$c_{min}$", y="Regret", hue="Approach", palette=palette, fliersize=0)
    # g.set(xscale="log")
    plt.yscale("log")
    plt.gcf().set_size_inches(5.5, 3.5)
    plt.tight_layout(pad=.5)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", "regret_over_cost_with_fixed_reward.pdf"))
    plt.show()


def plot_rho_sensitivity(df: pd.DataFrame):
    df.sort_values(by="approach")
    df["relevant"] = df["normalized budget"] == 1.0
    df = df.loc[df["relevant"]]
    m_ucb_end_regret = df.loc[df["approach"] == "m-UCB"].groupby("k").mean()
    bts_end_regret = df.loc[df["approach"] == "BTS"].groupby("k").mean()
    df = df.loc[df["our_approach"]]
    df[rho"$\rho$"] = df["approach"].apply(lambda x: extract_rho(x))
    g = sns.relplot(data=df, x=rho"$\rho$", y="regret", kind="line",
                    height=cm2inch(4.5)[0], aspect=1.5,
                    hue="k", col="p-min", col_wrap=2, facet_kws={"sharey": None},
                    markers="o", err_style="bars")
    g.set(xscale="log")
    g.set(yscale="log")
    # for mucb_val, bts_val in zip(m_ucb_end_regret, bts_end_regret):
    #     plt.axhline(mucb_val, color="green", label="m-UCB regret", zorder=0)
    #     plt.axhline(bts_val, color="red", label="BTS regret", zorder=0)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", "rho_sensitivity.pdf"))
    plt.show()


if __name__ == '__main__':
    use_results = True
    plot_results = True
    directory = os.path.join(os.getcwd(), "..", "results")
    filename = "bandit_comparison_full"
    filepath = os.path.join(directory, filename + ".csv")
    assert os.path.exists(directory)
    if not use_results:
        c_min = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]  # the setting with 1.0 is the traditional bandit setting.
        ks = [100, 30, 10]
        steps = 1e5  # we should be able to pull the cheapest arm 100000 times
        reps = 100
        dfs = []
        for k in tqdm(ks, desc="k"):
            for c in tqdm(c_min, leave=False, desc="c_min"):
                all_args = []
                process_start = process_time_ns()
                for rep in range(reps):
                    mean_rewards, mean_costs = create_setting(k, seed=rep, high_variance=hv, c_min=c)
                    mean_rewards, mean_costs = sort_setting(mean_rewards, mean_costs)
                    lowest_cost = np.min(mean_costs)
                    B = int(np.ceil(steps * lowest_cost))
                    args_list = [[b, int(steps), B, mean_rewards, mean_costs, rep, hv, c]
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
        df = pd.read_csv(filepath)
        df = prepare_df(df, every_nth=1)

        plot_rho_sensitivity(df.loc[df["our_approach"]])

        df = df.loc[df["approach"] != "w-UCB (a, rho=2)"]
        df = df.loc[df["approach"] != "w-UCB (a, rho=3)"]
        df = df.loc[df["approach"] != "w-UCB (a, rho=4)"]
        df = df.loc[df["approach"] != "UCB-SC+"]
        df = df.loc[df["approach"] != "i-UCB"]
        df = df.loc[df["approach"] != "c-UCB"]
        plot_regret_over_k(df)
        plot_regret(df, filename + ".pdf")
