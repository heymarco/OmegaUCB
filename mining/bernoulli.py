import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from util import load_df, prepare_df2, cm2inch, create_palette, move_legend_below_graph
from components.bandit_logging import *


import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = rho'\usepackage{times}'
mpl.rcParams['text.latex.preamble'] = rho'\usepackage{nicefrac}'
mpl.rc('font', family='serif')


def compute_ylims(df: pd.DataFrame, x, hue, col_var, x_cut=0.3):
    lims = []
    df = df.groupby([x, hue, col_var]).mean().reset_index()
    df = df[df[x] <= x_cut]
    df.sort_values(by=[col_var], inplace=True)
    for _, row_df in df.groupby(col_var):
        max_regret = row_df[NORMALIZED_REGRET].max()
        lims.append((0, max_regret))
    return lims


def plot_regret(df: pd.DataFrame):
    x = NORMALIZED_BUDGET
    y = NORMALIZED_REGRET
    hue = APPROACH
    col = K
    lims = compute_ylims(df, x, hue, col_var=col)
    palette = create_palette(df)
    df = df.sort_values(by=APPROACH_ORDER)
    g = sns.relplot(data=df, x=x, y=y, hue=hue, col=col,
                    kind="line", palette=palette,
                    facet_kws={"sharey": False}, err_style="bars")
    # g.set(yscale="log")
    g.set(xscale="log")
    for i, (lim, ax) in enumerate(zip(lims, g.axes.flatten())):
        ax.set_ylim(lim)
        if i > 0:
            ax.set_ylabel("")
    plt.gcf().set_size_inches(cm2inch(24, 6))
    plt.tight_layout(pad=.5)
    plt.subplots_adjust(right=.82, wspace=.22)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", "synth_bernoulli.pdf"))
    plt.show()


if __name__ == '__main__':
    filename = "uniform_vary_costs"
    df = load_df(filename)
    df = prepare_df2(df, n_steps=10)
    # df = df.loc[df[APPROACH] != "w-UCB (a, rho=1)"]
    df = df.loc[df[APPROACH] != "w-UCB (rho=1/5)"]
    df = df.loc[df[APPROACH] != "w-UCB (rho=1/6)"]
    df = df.loc[df[APPROACH] != "w-UCB (rho=2)"]
    df = df.loc[df[APPROACH] != "w-UCB (rho=3)"]
    df = df.loc[df[APPROACH] != "w-UCB (rho=4)"]
    # df = df.loc[df[APPROACH] != "UCB-SC+"]
    plot_regret(df)
