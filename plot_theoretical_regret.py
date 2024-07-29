import os.path

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from util import cm2inch

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}\usepackage[helvet]{sfmath}\everymath={\sf}'
mpl.rc('font', family='sans-serif')

sns.set_style("ticks")

if __name__ == '__main__':
    r1 = 0.5
    r2 = 0.5
    c1 = 0.25
    c2 = 0.5

    eta = 1.0
    rho = [0.25, 0.5, 1.0, 2.0]

    suboptimality = r1 / c1 - r2 / c2
    c_min = c1

    def compute_delta(subopt: float, c):
        return subopt / (subopt + 1 / c)

    def n_star(t: np.ndarray,
               rho: float,
               delta: float,
               mu_r: float,
               mu_c: float,
               eta = 1.0):
        a = eta * mu_r / (1 - mu_r)
        b = eta * (1 - mu_c) / mu_c
        return 8 * rho * np.log(t) / delta ** 2 * np.maximum(a, b)

    def xi_for_rho_equals_1(t: np.ndarray,
                            n_arms: int,
                            rho: float  # only there for consistency
                            ):
        assert rho == 1.0
        return (t - n_arms) * (1 - np.sqrt(1 - t ** -rho)) + (n_arms + 1) ** -rho + np.log(t) - np.log(n_arms + 1)

    def xi_for_rho_neq_1(t: np.ndarray,
                        n_arms: int,
                        rho: float):
        a = (t - n_arms) * (1 - np.sqrt(1 - t ** -rho))
        b = (n_arms + 1) ** -rho + 1 / (1 - rho) * (t ** (1 - rho)- (n_arms + 1) ** (1 - rho))
        return a + b

    def suboptimal_plays(t: np.ndarray,
               rho: float,
               delta: float,
               mu_r: float,
               mu_c: float,
               eta = 1.0):
        K = 2
        xi = xi_for_rho_equals_1 if rho == 1.0 else xi_for_rho_neq_1
        return 1 + n_star(t, rho, delta, mu_r, mu_c, eta) + xi(t, K, rho)

    def compute_regret(t: np.ndarray,
               rho: float,
               mu_1r: float,
               mu_1c: float,
               mu_2r: float,
               mu_2c: float,
               eta = 1.0):
        subopt = mu_1r / mu_1c - mu_2r / mu_2c
        const = 2 * mu_1r / mu_1c
        delta = compute_delta(subopt, mu_2c)
        return const + subopt * suboptimal_plays(t, rho, delta, mu_2r, mu_2c, eta)


    t = 1.3 ** np.arange(1, 60)
    results = []
    for r in rho:
        regret = compute_regret(t, r, r1, c1, r2, c2, eta)
        regret = np.expand_dims(regret, 0)
        regret = np.repeat(regret, 4, axis=0)
        regret[1] = t
        regret[2] = r
        regret[3] = t * c1 / 2

        results.append(regret.T)

    results = np.vstack(results)
    results = pd.DataFrame(results, columns=["regret", "t", r"$\rho$", "Budget"])

    g = sns.relplot(results, x="t", y="regret", hue=r"$\rho$", kind="line", style=r"$\rho$")
    plt.xscale("log")
    plt.ylim((0, 1200))
    plt.xlim((2, 10 ** 5))
    plt.gcf().set_size_inches(cm2inch((15, 4.8)))
    plt.tight_layout(pad=.7)
    plt.subplots_adjust(right=.83)
    ax = g.axes.flatten()[0]
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(.9)
    ax.set_ylabel("Regret upper bound")
    ax.set_xlabel("Number of plays until budget is exhausted")

    plt.savefig(os.path.join(os.getcwd(), "figures", "theoretical_regret.pdf"))
    plt.show()

