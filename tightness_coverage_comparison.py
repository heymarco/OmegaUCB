import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from matplotlib import pyplot as plt
import seaborn as sns

from approach_names import *
from util import cm2inch

sns.set_style("ticks")

import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')


def wilson_generalized(mu, n, z, eta=1.0, m=0.0, M=1.0):
    """
    Returns the lower and upper confidence bound of our confidence interval
    :param mu: sample mean
    :param n: sample size
    :param z: number of standard deviations
    :param eta: variance scaling parameter
    :param m: lower bound of random variable
    :param M: upper bound of random variable
    :return: lower and upper confidence bound
    """
    K = eta * (z ** 2)
    A = (n + K)
    B = (2 * n * mu + K * (M + m))
    C = (n * mu ** 2 + K * M * m)
    lcb = B / (2 * A) - np.sqrt((B / (2 * A)) ** 2 - C / A)
    ucb = B / (2 * A) + np.sqrt((B / (2 * A)) ** 2 - C / A)
    return lcb, ucb


def hoeffding_ci(n, delta: float):
    """
    Computes half the width of the confidence interval for a random variable in [0,1] using Hoeffding's inequality
    :param n: sample size
    :param delta: confidence level
    :return: half the width of the confidence interval
    """
    return np.sqrt(-1 / (2 * n) * np.log(delta / 2))


# Confidence bounds for our method and its competitors

def our_method(mu_r, mu_c, n, delta=0.05, eta=1.0, m=0, M=1):
    z = norm.interval(1 - delta / 2)[1]
    lcb_cost = wilson_generalized(mu_c, n, z, eta, m, M)[0]
    ucb_rew = wilson_generalized(mu_r, n, z, eta, m, M)[1]
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
    return top / max(0.001, bottom)  # see eq 2 in the respective paper


def ucb_b2(mu_r, mu_c, n, delta=0.05):
    r_hat = mu_r / np.maximum(mu_c, 1e-10)
    vr = (1 - mu_r) * mu_r
    vc = (1 - mu_c) * mu_c
    x = -np.log(delta / 2)
    eps = np.sqrt(2 * vr * x / n) + 3 * x / n
    eta = np.sqrt(2 * vc * x / n) + 3 * x / n
    c = 1.4 * (eps + r_hat * eta) / np.maximum(mu_c, 1e-10)
    ucb = r_hat + c
    return ucb


def evaluate_once(approaches: dict, exp_c, n, seed, delta=0.05):
    """
    Performs one evaluation of the approaches
    on a bernoulli cost distribution with known expected value exp_c and
    randomly parameterized reward distribution
    :param approaches: dictionary mapping the name of the approach to a function that computes the ucb for the reward-cost ratio
    :param exp_c: expected cost
    :param n: sample size
    :param seed: random seed
    :param delta: confidence level (values closer to 0 -> more confident)
    :return: list of lists where each row corresponds to the results of the approach on the sampled data. The columns represent name, sample size, expected rewards, expected costs, and the ucb
    """
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
    # If the figure should be narrow or wide
    narrow = True
    # sample sizes
    ns = [100, 1000, 10000, 100000]
    # number of repetitions (number that evaluate_once is called)
    repetitions = 10000
    # confidence level
    delta = 0.01
    # will hold the results
    results = None

    # define the column strings
    APPROACH, SAMPLES, MU_R, MU_C, UCB = "Approach", "Samples", r"$\mu_r$", r"$\mu_c$", "UCB"
    EXP_VALUE_RATIO, UCB_EXPECTATION, UCB_VIOLATIONS = "exp. ratio", "UCB/expect.", r"\% UCB viol."
    columns = [APPROACH, SAMPLES, MU_R, MU_C, UCB]

    # define the names of the approaches as they should appear in the figure
    approaches = {
        OMEGA_UCB_ + " (c, ours)": our_method,
        CUCB + " (h)": c_ucb,
        MUCB + " (c)": m_ucb,
        IUCB + " (u)": i_ucb,
        BUDGET_UCB + " (h)": b_ucb,
        UCB_SC + " (u)": ucb_sc,
        UCB_B2_name + " (u)": ucb_b2,
    }
    # define the order in which the approaches should appear in the figure
    order = {
        OMEGA_UCB_ + " (c, ours)": 1,
        MUCB + " (c)": 2,
        BUDGET_UCB + " (h)": 3,
        CUCB + " (h)": 4,
        IUCB + " (u)": 5,
        UCB_SC + " (u)": 6,
        UCB_B2_name + " (u)": 7,
    }

    # for all sample sizes, for all experiment repetitions:
    # evaluate the approaches and append the results to the 'result' list
    cost_rng = np.random.default_rng(seed=0)
    for n in ns:
        for rep in range(repetitions):
            exp_c = cost_rng.uniform()
            result = evaluate_once(approaches, exp_c, n, seed=rep + 1, delta=delta)
            if results is None:
                results = result
            else:
                results += result
    # create the data frame from the results
    df = pd.DataFrame(results, columns=columns)

    # compute the metrics shown in the figure
    df[EXP_VALUE_RATIO] = np.nan
    df[EXP_VALUE_RATIO] = df[MU_R] / df[MU_C]
    df["UCB > exp. ratio"] = np.nan
    df["Coverage"] = df[UCB] > df[EXP_VALUE_RATIO]
    df[UCB_VIOLATIONS] = df[UCB] < df[EXP_VALUE_RATIO]
    df[UCB_EXPECTATION] = np.nan
    df[UCB_EXPECTATION] = df[UCB] / df[EXP_VALUE_RATIO]
    df[UCB_VIOLATIONS] = df[UCB_VIOLATIONS] * 100
    df["order"] = np.nan
    for approach in order:
        df.loc[df[APPROACH] == approach, "order"] = order[approach]

    df = pd.melt(df, id_vars=[APPROACH, SAMPLES, "order"],
                 value_vars=[UCB_VIOLATIONS, UCB_EXPECTATION],
                 value_name="Score", var_name="Metric")
    df = df.sort_values(by="order")

    sns.set_palette(sns.cubehelix_palette(n_colors=len(ns)))
    if narrow:
        g = sns.catplot(data=df, kind="bar", x=APPROACH, y="Score",
                        hue="Samples", row="Metric",
                        sharex=True, sharey=False, errwidth=1.0)
    else:
        g = sns.catplot(data=df, kind="bar", x=APPROACH, y="Score",
                        hue="Samples", col="Metric",
                        sharex=True, sharey=False, errwidth=1.0)
    g.axes.flatten()[1].set_yscale("log")
    g.axes.flatten()[1].set_ylim(bottom=0.5)
    hlines_ucb = [1, 10, 100]
    hlines_cov = [5, 10, 15]
    for hline in hlines_ucb:
        g.axes.flatten()[1].axhline(hline, ls="--", lw=0.7, color="black", zorder=0)
    for hline in hlines_cov:
        g.axes.flatten()[0].axhline(hline, ls="--", lw=0.7, color="black", zorder=0)
        g.axes.flatten()[0].set_yticks([5, 10, 15], ["5", "10", "15"])
    g.axes.flatten()[0].set_ylim(0, 18)
    g.set(xlabel=None)
    for ax_index, ax in enumerate(g.axes.flatten()):
        title = ax.get_title()
        title = title.split(" = ")[-1]
        ax.set_ylabel(title)
        ax.set_title("")
        if narrow and ax_index == 1:
            plt.xticks(rotation=18, ha="right")
        if not narrow:
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=30, ha='right')
    if narrow:
        sns.move_legend(g, "upper center", ncol=4, frameon=True)
        plt.gcf().set_size_inches(cm2inch((10, 7.8)))
        plt.tight_layout(pad=.5)
        plt.subplots_adjust(top=.8)
        plt.savefig(os.path.join("figures", "ucb_comparison_narrow.pdf"))
    else:
        plt.gcf().set_size_inches(cm2inch((15, 4)))
        plt.tight_layout(pad=.5)
        plt.subplots_adjust(right=.82, wspace=.3, top=0.93)
        sns.move_legend(g, "upper right")
        plt.savefig(os.path.join("figures", "ucb_comparison.pdf"))
    plt.show()
