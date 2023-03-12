import os

import numpy as np
import pandas as pd
from scipy.stats import  norm
from matplotlib import pyplot as plt
import seaborn as sns

from approach_names import *
from util import cm2inch, move_legend_below_graph

sns.set_style("ticks")

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
mpl.rc('font', family='serif')


def wilson_generalized(x, n, z, eta=1.0, m=0.0, M=1.0):
    K = eta * (z ** 2)
    A = (n + K)
    B = (2 * n * x + K * (M + m))
    C = (n * x ** 2 + K * M * m)
    lcb = B / (2 * A) - np.sqrt((B / (2 * A)) ** 2 - C / A)
    ucb = B / (2 * A) + np.sqrt((B / (2 * A)) ** 2 - C / A)
    return lcb, ucb


def hoeffding_ci(n, delta: float):
    return np.sqrt(-1/(2 * n) * np.log(delta / 2))


def our_method(mu_r, mu_c, n, delta=0.05, eta=1.0, m=0, M=1):
    z = norm.interval(1 - delta / 2)[1]
    lcb_cost = wilson_generalized(mu_c, n, z, eta, m, M)[0]
    ucb_rew = wilson_generalized(mu_r, n, z, eta, m, M)[1]
    if lcb_cost <= 0:
        print("error")
    return ucb_rew / lcb_cost


def m_ucb(mu_r, mu_c, n, delta=0.05):
    ucb = mu_r + hoeffding_ci(n, delta)
    lcb = mu_c - hoeffding_ci(n, delta)
    return min(ucb, 1) / max(lcb, 0.001)


def i_ucb(mu_r, mu_c, n, delta=0.05):
    ci = hoeffding_ci(n, delta)
    return mu_r / mu_c + ci


def c_ucb(mu_r, mu_c, n, delta=0.05):
    ci = hoeffding_ci(n, delta)
    return (mu_r + ci) / mu_c


def b_ucb(mu_r, mu_c, n, delta=0.05):
    ci = hoeffding_ci(n, delta)
    term1 = mu_r / mu_c
    term2 = ci / mu_c
    term3 = term2 * min(mu_r + ci, 1) / max(mu_r - ci, 0.001)
    return term1 + term2 + term3


def ucb_sc(mu_r, mu_c, n, delta=0.05):
    top = mu_r * mu_c + np.sqrt(delta / 2 * (mu_r ** 2 + mu_c ** 2 - delta / 2))
    bottom = mu_c ** 2 - delta / 2
    return top / max(0.001, bottom)


def evaluate_once(approaches: dict, exp_c, n, seed, delta=0.05):
    rng = np.random.default_rng(seed)
    exp_r = rng.uniform()
    rewards = rng.uniform(size=n) < exp_r
    avg_c = 0
    while avg_c == 0.0:
        costs = rng.uniform(size=n) < exp_c
        avg_c = np.mean(costs)
    avg_r = np.mean(rewards)
    result = []
    for name, approach in approaches.items():
        ucb = approach(avg_r, avg_c, n, delta)
        result.append([
            name, n, exp_r, exp_c, ucb
        ])
    return result


if __name__ == '__main__':
    ns = [100, 1000, 10000, 100000]
    repetitions = 10000
    alpha = 0.01
    results = None
    columns = ["Approach", "Samples", r"$\mu_r$", r"$\mu_c$", "UCB"]
    approaches = {
        OMEGA_UCB_ + " (c, ours)": our_method,
        CUCB + " (u)": c_ucb,
        MUCB + " (c)": m_ucb,
        IUCB + " (u)": i_ucb,
        BUDGET_UCB + " (c)": b_ucb,
        UCB_SC + " (u)": ucb_sc
    }
    order = {
        OMEGA_UCB_ + " (c, ours)": 1,
        CUCB + " (u)": 5,
        MUCB + " (c)": 2,
        IUCB + " (u)": 4,
        BUDGET_UCB + " (c)": 3,
        UCB_SC + " (u)": 6
    }
    cost_rng = np.random.default_rng(seed=0)
    for n in ns:
        for rep in range(repetitions):
            exp_c = cost_rng.uniform()
            result = evaluate_once(approaches, exp_c, n, seed=rep + 1, delta=alpha)
            if results is None:
                results = result
            else:
                results += result

    df = pd.DataFrame(results, columns=columns)
    df["exp. ratio"] = np.nan
    df["exp. ratio"] = df[r"$\mu_r$"] / df[r"$\mu_c$"]
    df["UCB > exp. ratio"] = np.nan
    df["Coverage"] = df["UCB"] > df["exp. ratio"]
    df["UCB (relative)"] = np.nan
    df["UCB (relative)"] = df["UCB"] / df["exp. ratio"]
    df = df.groupby(["Samples", r"$\mu_c$", "Approach"]).mean().reset_index()
    df["order"] = np.nan
    for approach in order:
        df["order"].loc[df["Approach"] == approach] = order[approach]

    df = pd.melt(df, id_vars=["Approach", "Samples", "order"], value_vars=["UCB (relative)", "Coverage"],
                 value_name="Score", var_name="Metric")
    df = df.sort_values(by="order")

    sns.set_palette(sns.cubehelix_palette(n_colors=len(ns)))
    g = sns.catplot(data=df, kind="bar", x="Approach", y="Score", hue="Samples", col="Metric",
                    sharex=True, sharey=False, errwidth=1.0)
    g.axes.flatten()[0].set_yscale("log")
    g.axes.flatten()[0].set_ylim(bottom=0.5)
    hlines_ucb = [1, 10, 100]  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100]
    hlines_cov = [0.8, 0.85, 0.9, 0.95, 1.0]
    for hline in hlines_ucb:
        g.axes.flatten()[0].axhline(hline, ls="--", lw=0.7, color="black", zorder=0)
    for hline in hlines_cov:
        g.axes.flatten()[1].axhline(hline, ls="--", lw=0.7, color="black", zorder=0)
    # g.axes.flatten()[0].set_ylim(0, 10.0)
    g.axes.flatten()[1].set_ylim(0.8, 1.0)
    g.set(xlabel=None)
    for ax in g.axes.flatten():
        title = ax.get_title()
        title = title.split(" = ")[-1]
        ax.set_ylabel(title)
        ax.set_title("")
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=30, ha='right')
    plt.gcf().set_size_inches(cm2inch((15, 4)))
    plt.tight_layout(pad=.5)
    plt.subplots_adjust(right=.82, wspace=.3)
    plt.savefig(os.path.join("figures", "ucb_comparison.pdf"))
    plt.show()

