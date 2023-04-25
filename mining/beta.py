import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from util import load_df, prepare_df, cm2inch, create_palette
from components.bandit_logging import *
from approach_names import *
from colors import get_markers_for_approaches


sns.set_style(style="ticks")

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
mpl.rc('font', family='serif')


def compute_ylims(df: pd.DataFrame, x, hue, col_var, x_cut=.3):
    lims = []
    df = df.groupby([x, hue, col_var]).mean().reset_index()
    df = df[df[x] <= x_cut]
    df.sort_values(by=[col_var], inplace=True)
    for _, row_df in df.groupby(col_var):
        max_regret = row_df[NORMALIZED_REGRET].max()
        min_regret = row_df[NORMALIZED_REGRET].min()
        lims.append((0, max_regret))
    return lims


def plot_regret(df: pd.DataFrame, with_ci: bool = False):
    df = df.iloc[::-1]
    x = NORMALIZED_BUDGET
    y = NORMALIZED_REGRET
    hue = APPROACH
    col = K
    # df = df.sort_values(by=[APPROACH])
    palette = create_palette(df)
    markers = get_markers_for_approaches(np.unique(df[APPROACH]))
    if with_ci:
        g = sns.relplot(data=df, x=x, y=y, hue=hue, col=col,
                        # lw=1, markersize=3,
                        markeredgewidth=0.1,
                        kind="line", palette=palette, legend=False,
                        errorbar="ci", err_style="bars", err_kws={"capsize": 2}, solid_capstyle="butt",
                        seed=0, n_boot=500,
                        facet_kws={"sharey": False},
                        # style=hue, markers=markers,
                        dashes=False)
    else:
        g = sns.relplot(data=df, x=x, y=y, hue=hue, col=col,
                        # lw=1, markersize=3,
                        markeredgewidth=0.1,
                        kind="line", palette=palette, legend=False,
                        errorbar=None,
                        facet_kws={"sharey": False},
                        style=hue, markers=markers,
                        dashes=False)
    # g.set(xscale="symlog")
    # g.set(linthreshx=0.01)
    lims = [(0, 0.009), (0, 0.09), (0, 0.22)]
    for i, (lim, ax) in enumerate(zip(lims, g.axes.flatten())):
        ax.set_ylim(lim)
        ax.set_xlim((0.095, 1))
        ax.set_xscale("symlog", linthresh=.1)
        if i > 0:
            ax.set_ylabel("")
    plt.gcf().set_size_inches(cm2inch(20, 6 * 0.75))
    # create_custom_legend(g)
    plt.tight_layout(pad=.8)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", filename + ".pdf"))
    plt.show()


if __name__ == '__main__':
    with_ci = True
    filename = "synth_beta_combined"
    df = load_df(filename)
    df = prepare_df(df, n_steps=10)
    df.loc[df[APPROACH] == "B-UCB", APPROACH] = BUDGET_UCB
    df = df.loc[df[APPROACH] != OMEGA_UCB_1_64]
    df = df.loc[df[APPROACH] != OMEGA_UCB_1_32]
    df = df.loc[df[APPROACH] != OMEGA_UCB_1_16]
    df = df.loc[df[APPROACH] != OMEGA_UCB_1_8]
    # df = df.loc[df[APPROACH] != OMEGA_UCB_1_4]
    df = df.loc[df[APPROACH] != OMEGA_UCB_1_2]
    # df = df.loc[df[APPROACH] != OMEGA_UCB_1]
    df = df.loc[df[APPROACH] != OMEGA_UCB_2]
    df = df.loc[df[APPROACH] != ETA_UCB_1_64]
    df = df.loc[df[APPROACH] != ETA_UCB_1_32]
    df = df.loc[df[APPROACH] != ETA_UCB_1_16]
    df = df.loc[df[APPROACH] != ETA_UCB_1_8]
    # df = df.loc[df[APPROACH] != ETA_UCB_1_4]
    df = df.loc[df[APPROACH] != ETA_UCB_1_2]
    # df = df.loc[df[APPROACH] != ETA_UCB_1]
    df = df.loc[df[APPROACH] != ETA_UCB_2]
    # df = df.loc[df[APPROACH] != UCB_SC_PLUS]
    df = df.loc[df[APPROACH] != BUDGET_UCB]
    # df = df.loc[df[APPROACH] != BTS]
    # df = df.loc[df[APPROACH] != B_GREEDY]
    # df = df.loc[df[APPROACH] != CUCB]
    # df = df.loc[df[APPROACH] != MUCB]
    # df = df.loc[df[APPROACH] != IUCB]

    plot_regret(df, with_ci=with_ci)
