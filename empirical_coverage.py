import os

import pandas as pd
import numpy as np
from scipy.stats import norm, truncnorm
from scipy.special import erf

import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib as mpl

from util import cm2inch
from components.bandit_logging import RHO

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
mpl.rc('font', family='serif')


def avg_prob(z):
    this_avg = truncnorm.mean(a=-np.infty, b=-z)
    avg_conf = 1 - norm.cdf(-this_avg)
    return avg_conf


def avg_prob_2(z, t, r):
    z_star = norm.pdf(z) / (1 - norm.cdf(z))
    z_star_approx = 1 / np.sqrt(2 * np.pi) * t ** (-r) / (1 - 1 / 2 * (1 + np.sqrt(1 - np.exp(-1 / 2 * z ** 2))))
    approx_3 = (np.exp(-z**2/2)*z / (4 * np.sqrt(1-np.exp(-z**2/2)))) / (1 - 1 / 2 * (1 + np.sqrt(1 - np.exp(-1 / 2 * z ** 2))))
    avg_conf = 1 - norm.cdf(z_star)
    # avg_conf_2 = 1 - norm.cdf(z_star_approx)
    avg_conf_2 = 1 - norm.cdf(approx_3)
    # ac3 = (1 - 1 / 2 * (1 + np.sqrt(1 - np.exp(-1 / 2 * approx_3 ** 2))))
    return avg_conf


def lcb_wilson_generalized(x, n, z, eta=1.0, m=0.0, M=1.0):
    K = eta * (z ** 2)
    A = (n + K)
    B = (2 * n * x + K * (M + m))
    C = (n * x ** 2 + K * M * m)
    lcb = B / (2 * A) - np.sqrt((B / (2 * A)) ** 2 - C / A)
    return lcb


def ucb_wilson_generalized(x, n, z, eta=1.0, m=0.0, M=1.0):
    K = eta * (z ** 2)
    A = (n + K)
    B = (2 * n * x + K * (M + m))
    C = (n * x ** 2 + K * M * m)
    ucb = B / (2 * A) + np.sqrt((B / (2 * A)) ** 2 - C / A)
    return ucb


def bernoulli_average(n, expected_val, rng):
    avg = []
    for this_n, this_exp in zip(n, expected_val):
        bern_samples = rng.uniform(size=this_n) < this_exp
        avg.append(bern_samples.mean())
    return np.array(avg)


if __name__ == '__main__':
    TIME = r"$t$"
    ID_ACHIEVED_CONF = r"Achieved conf."
    ID_DESIRED_CONF = r"Min. conf. for log. regret"
    ID_NOM_CONF = r"Theoretical conf."
    ID_TRUE_NOM_CONF = r"$1 - \mathrm{erf}\left(\sqrt{\rho\log t}\right)$"
    n_tries = 10000
    rho = [2.0] + [2.0 ** (-x) for x in range(9)]
    rho_test = np.arange(1, 101) / 100
    max_steps = [1 * 10 ** x for x in range(1, 6)]
    result = []
    # for rep in range(reps):
    rng = np.random.default_rng(0)
    for n_steps in max_steps:
        t = np.ones(shape=n_tries) * n_steps
        n = rng.integers(1, 1 + t)
        exp_rewards = rng.uniform(size=t.shape)
        exp_costs = rng.uniform(size=t.shape)
        avg_rewards = bernoulli_average(n, exp_rewards, rng)
        avg_costs = bernoulli_average(n, exp_costs, rng)
        for r in rho_test:
            z = np.sqrt(2 * r * np.log(n_steps))
            p_ab = 0  # 1 / 4 * (1 - np.sqrt(1 - n_steps ** (-r))) ** 2
            p_ab_true = 0  # 1 / 4 * (1 - erf(z / np.sqrt(2))) ** 2
            conf = np.sqrt(1 - n_steps ** (-1.0))
            nom_conf = 1 - (1 - np.sqrt(1 - n_steps ** (-r)) - p_ab)
            nom_conf_2 = 1 - (1 - erf(z / np.sqrt(2)) - p_ab_true)
            if n_steps == 100 and r == 0.25:
                print(nom_conf_2 - nom_conf)
            # nom_conf = 1 - (p_ab + p_mu_c_large_enough + p_mu_r_large_enough)
            true_z = norm.interval(conf)[1]
            true_nom_conf = 1 - (1 - erf(z / np.sqrt(2)) - p_ab_true)
            ucb = ucb_wilson_generalized(avg_rewards, n, z, eta=1)
            lcb = lcb_wilson_generalized(avg_costs, n, z, eta=1)
            for u, l, er, ec in zip(ucb, lcb, exp_rewards, exp_costs):
                result.append([r, n_steps, z, conf, nom_conf, true_nom_conf,
                               u, l, er, ec])
    columns = [RHO, TIME, "z", ID_DESIRED_CONF, ID_NOM_CONF, ID_TRUE_NOM_CONF,
               "UCB", "LCB", "Exp. rew.", "Exp. cost"]
    df = pd.DataFrame(result, columns=columns)
    df["UCB (ratio)"] = df["UCB"] / df["LCB"]
    df["Ratio"] = df["Exp. rew."] / df["Exp. cost"]
    df[ID_ACHIEVED_CONF] = df["UCB (ratio)"] > df["Ratio"]
    print(df)
    df = pd.melt(df, id_vars=[RHO, TIME], value_vars=[ID_DESIRED_CONF, ID_NOM_CONF,
                                                      # ID_TRUE_NOM_CONF,
                                                      ID_ACHIEVED_CONF],
                 var_name="Type", value_name="Confidence")

    g = sns.relplot(data=df, kind="line", x=RHO, y="Confidence", col=TIME , hue="Type",
                    errorbar=None,
                    # facet_kws={"sharey": False},
                    style="Type",
                    palette=["black",
                             sns.color_palette("RdBu", n_colors=5)[0],
                             # "#cab873",
                             sns.color_palette("RdBu_r", n_colors=5)[0]],
                    dashes={ID_DESIRED_CONF: (2, 2), ID_ACHIEVED_CONF: "",
                            ID_NOM_CONF: (1, 1), ID_TRUE_NOM_CONF: (1, 2)},
                    lw=1.5
                    )
    # plt.xscale("log")
    for t, ax in zip(max_steps, g.axes.flatten()):
        ax.set_ylim((0.55, 1.01))
        ax.set_xlim((.01, .7))
        log_conf = np.sqrt(1 - 1 / t)
        for r in rho:
            ax.axvline(r, lw=1.5 if r == 1 / 4 else .5, color="lightgray", zorder=0)
    sns.move_legend(
        g,
        loc="upper center",
        # bbox_to_anchor=(0, 0.72, 1, 0.2),
        mode="expand",
        borderaxespad=1,
        ncol=4,
        title="",
        frameon=True,
    )
    plt.gcf().set_size_inches(cm2inch((15, 5.5)))
    plt.tight_layout(pad=.5)
    plt.subplots_adjust(top=.65, hspace=.5)
    plt.savefig(os.path.join("figures", "rho_investigation.pdf"))
    plt.show()