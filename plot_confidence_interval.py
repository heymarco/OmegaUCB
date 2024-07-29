import os

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



    n_steps = 100
    sample_mean = np.linspace(0, 1, n_steps)

    etas = [1, 0.1, 0.01]
    z = 3
    n = 30

    data = []
    for eta in etas:
        lcb, ucb = wilson_generalized(sample_mean, n, z, eta)
        results_for_eta = np.vstack([sample_mean, sample_mean, lcb, ucb, np.repeat(eta, len(ucb))])
        data.append(results_for_eta.T)

    data = np.vstack(data)
    SAMPLE_MEAN = "sample mean"
    MU = "mu"
    LCB = "lcb"
    UCB = "ucb"
    ETA = r"$\eta$"
    df_cols = [SAMPLE_MEAN, MU, LCB, UCB, ETA]
    col_order = {eta: i for i, eta in enumerate(etas)}
    palette = {LCB: "gray", UCB: "gray", MU: "black"}
    df = pd.DataFrame(data, columns=df_cols)

    df = df.melt(value_vars=df_cols[1:-1], id_vars=[SAMPLE_MEAN, ETA], var_name="quantity")
    g = sns.relplot(df, kind="line", x=df_cols[0], y="value",
                    hue="quantity", col=ETA, col_order=col_order,
                    palette=palette, linewidth=.7)

    for ax_pos, ax in enumerate(g.axes.flatten()):
        line = ax.get_lines()
        ax.fill_between(line[0].get_xdata(), line[1].get_ydata(), line[2].get_ydata(), color='black', alpha=.1)
        ax.set_xlabel("" if ax_pos != 1 else r"sample mean ($n=20$)")
        ax.set_ylabel(r"sample mean and CI")
        # ax.axhline(1.0, 0.0, color="gray", zorder=0)
        # ax.axvline(1.0, 0.0, color="gray", zorder=0)
        ax.set_ylim((0, 1))
        ax.set_xlim((0, 1))
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth(.9)

    if g.legend:
        g.legend.remove()

    plt.gcf().set_size_inches(cm2inch((15, 5.5)))
    plt.tight_layout(pad=.7)
    plt.savefig(os.path.join(os.getcwd(), "figures", "confidence_interval.pdf"))
    plt.show()
