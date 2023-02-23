import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from util import load_df, prepare_df, cm2inch, create_palette, move_legend_below_graph, create_custom_legend
from components.bandit_logging import *
from approach_names import *

sns.set_style(style="ticks")

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
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
    df = df.sort_values(by=[APPROACH])
    palette = create_palette(df)
    g = sns.relplot(data=df, x=x, y=y, hue=hue, col=col,
                    kind="line", palette=palette, legend=False,
                    facet_kws={"sharey": False}, err_style="bars")
    # g.set(yscale="log")
    g.set(xscale="log")
    for i, (lim, ax) in enumerate(zip(lims, g.axes.flatten())):
        ax.set_ylim(lim)
        if i > 0:
            ax.set_ylabel("")
        if ax.get_legend():
            sns.move_legend(ax, "upper center", bbox_to_anchor=(1, 1))
    plt.gcf().set_size_inches(cm2inch(20, 6))
    create_custom_legend(g)
    plt.tight_layout(pad=.7)
    plt.subplots_adjust(top=0.65)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", "synth_bernoulli.pdf"))
    plt.show()


if __name__ == '__main__':
    filename = "synth_bernoulli"
    df = load_df(filename)
    df = prepare_df(df, n_steps=10)
    df = df.loc[df[APPROACH] != OMEGA_UCB_1_32]
    df = df.loc[df[APPROACH] != OMEGA_UCB_1_16]
    df = df.loc[df[APPROACH] != OMEGA_UCB_1_8]
    # df = df.loc[df[APPROACH] != OMEGA_UCB_1_4]
    df = df.loc[df[APPROACH] != OMEGA_UCB_1_2]
    # df = df.loc[df[APPROACH] != OMEGA_UCB_1]
    df = df.loc[df[APPROACH] != OMEGA_UCB_2]
    plot_regret(df)
