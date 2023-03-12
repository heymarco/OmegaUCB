import os.path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import norm

import matplotlib as mpl

from util import cm2inch

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
mpl.rc('font', family='serif')


def ci_hoeffding(x, delta, n):
    ci = np.sqrt(1 / (2 * n) * np.log(2 / delta))
    return x - ci, x, x + ci


def ci_wilson(x, delta, n, eta=1.0):
    z = norm.interval(1 - delta)[1]
    m, M = 0, 1
    K = eta * (z ** 2)
    A = (n + K)
    B = (2 * n * x + K * (M + m))
    C = (n * x ** 2 + K * M * m)
    low = B / (2 * A) - np.sqrt((B / (2 * A)) ** 2 - C / A)
    high = B / (2 * A) + np.sqrt((B / (2 * A)) ** 2 - C / A)
    return low, x, high


if __name__ == '__main__':
    delta = 0.01
    x = np.array([0.1, 0.5, 0.9])
    n = 100

    plt.axvline(0.0, ls="--", lw=0.7, color="black")
    plt.axvline(1.0, ls="--", lw=0.7, color="black")

    funcs = {
        "Asymmetric confidence intervals (our method)": ci_wilson,
        "Symmetric confidence intervals (Hoeffding)": ci_hoeffding
    }
    color = sns.cubehelix_palette(n_colors=4)[1]
    plt.axvspan(-1, 0, alpha=0.5, color=color)
    plt.axvspan(1, 2, alpha=0.5, color=color)
    for i, func_name in enumerate(funcs):
        label = func_name
        func = funcs[func_name]
        low, mean, high = func(x, delta, n)
        xlow = np.abs(x-low)
        xhigh = np.abs(x-high)
        plt.scatter(x=x, y=np.ones(len(x)) * i, color="black", s=12)
        plt.errorbar(x=x, y=np.ones(len(x)) * i, xerr=[xlow, xhigh],
                     lw=0, elinewidth=1, capsize=3, color="black")

    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.5, 1.5)
    # plt.xlabel("Cost")
    plt.yticks(np.arange(len(funcs)), funcs.keys())
    plt.xticks([0, 1], [0, 1])
    plt.gcf().set_size_inches(cm2inch((14, 1.8)))
    plt.tight_layout(pad=0.13)
    plt.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(os.getcwd(), "figures", "comparison_ci_hoeffding_wilson.pdf"))
    plt.show()


