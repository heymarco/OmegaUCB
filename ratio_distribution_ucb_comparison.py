import os

import numpy as np
from scipy.stats import  norm
from matplotlib import pyplot as plt
import seaborn as sns

from util import cm2inch

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{times}'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{nicefrac}'
mpl.rc('font', family='serif')

if __name__ == '__main__':
    n_steps = 1000
    x = np.arange(1, n_steps + 1) / n_steps
    n = 1000
    delta = 0.05

    rng = np.random.default_rng(0)

    samples = [
        rng.uniform(size=n) < x_val for x_val in x
    ]
    samples = np.array(samples)
    samples = np.mean(samples, axis=1)

    def lcb_hoeffding(x, n, alpha=1):
        return np.maximum(x - alpha * np.sqrt(-np.log(delta) / (2 * n)), 0)

    def lcb_wilson(x, n):
        z = norm.interval(1 - delta)[1]
        ns = x * n
        nf = n - ns
        z2 = np.power(z, 2)
        avg = (ns + 0.5 * z2) / (n + z2)
        ci = z / (n + z2) * np.sqrt((ns * nf) / n + z2 / 4)
        return avg - ci

    relative_ci_width_hoeffding = (1 / lcb_hoeffding(x, n) - 1 / x) / (1 / x)
    relative_ci_width_wilson = (1 / lcb_wilson(x, n) - 1 / x) / (1 / x)

    color_mean = sns.color_palette("Greys", n_colors=3)[1]
    color_hoeffding = sns.color_palette("Greens", n_colors=3)[1]
    color_wilson = sns.color_palette("Blues", n_colors=3)[2]

    plt.plot(x, relative_ci_width_hoeffding, label="Symmetric CI (Hoeffding)", color=color_hoeffding)
    plt.plot(x, relative_ci_width_wilson, label="Asymmetric CI (our method)", color=color_wilson)

    plt.legend()
    plt.xscale("symlog")
    plt.yscale("symlog")
    plt.xlabel(r"Mean $\overline x$ of {} samples".format(n))
    plt.ylabel(r"Relative size of CI (below $\overline x$)")
    plt.gcf().set_size_inches(cm2inch((12, 5.5)))
    plt.tight_layout(pad=.5)
    plt.savefig(os.path.join(os.getcwd(), "figures", "ratio_dist_ci_comparison.pdf"))
    plt.show()

    lcb_hoeffding_inv_emp = 1 / lcb_hoeffding(samples, n)
    lcb_wilson_inv_emp = 1 / lcb_wilson(samples, n)
    lcb_wilson_inv = 1 / lcb_wilson(x, n)
    lcb_hoeffding_inv = 1 / lcb_hoeffding(x, n)

    plt.plot(x, 1 / x, color=color_mean, label=r"Expected value", ls="--")
    plt.plot(x, 1 / samples, color=color_mean, label=r"Sample mean")
    plt.plot(x, lcb_hoeffding_inv, label="Symmetric CI (Hoeffding)", color=color_hoeffding, ls="--")
    plt.plot(x, lcb_wilson_inv, label="Asymmetric CI (our method)", color=color_wilson, ls="--")
    # plt.plot(x, lcb_hoeffding_inv_emp, color=color_hoeffding)
    # plt.plot(x, lcb_wilson_inv_emp, color=color_wilson)

    plt.xlabel(r"Average cost of the arm (1000 samples)".format(n))
    plt.ylabel(r"UCB of reward-cost ratio")
    plt.gcf().set_size_inches(cm2inch((12, 6.5)))
    plt.xlim(left=0.0, right=0.15)
    plt.ylim(bottom=4)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "figures", "ratio_dist_lcb_comparison.pdf"))
    plt.show()
