import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from colors import get_markers_for_approaches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from util import load_df, prepare_df, cm2inch, create_palette, create_custom_legend
from components.bandit_logging import *
from approach_names import *

sns.set_style(style="ticks")

import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
mpl.rc('font', family='serif')


def get_upper_percentile_data(df: pd.DataFrame):
    df = (df.groupby([APPROACH, NORMALIZED_BUDGET])
          .apply(lambda gdf: gdf[gdf[REGRET] > np.quantile(gdf[REGRET], 0.75)])
          )
    return df


def compute_ylims(df: pd.DataFrame, x, hue, col_var, x_cut=.2):
    lims = []
    df = df.groupby([x, hue, col_var]).mean().reset_index()
    df = df[df[x] <= x_cut]
    for _, row_df in df.groupby(col_var):
        max_regret = row_df[REGRET].max()
        lims.append((0, max_regret))
    return lims


def plot_regret(df: pd.DataFrame, filename: str, with_ci: bool = False):
    sns.set()
    x = NORMALIZED_BUDGET
    y = REGRET
    hue = APPROACH
    col = K
    lims = compute_ylims(df, x, hue, col_var=col)
    df = df.iloc[::-1]
    palette = create_palette(df)
    markers = get_markers_for_approaches(np.unique(df[APPROACH]))
    df = df[df[K] == 100]
    # df = get_upper_percentile_data(df)
    data_rho_1 = df[df[APPROACH] == OMEGA_UCB_1]
    data_rho_14 = df[df[APPROACH] == OMEGA_UCB_1_4]
    palette = sns.color_palette(n_colors=2)
    ax = sns.lineplot(x=data_rho_1[NORMALIZED_BUDGET].to_numpy(),
                      y=data_rho_1[REGRET].to_numpy(),
                      color=palette[0],
                      estimator="mean",
                      err_style="bars",
                      errorbar=None,  # ("pi", 100),
                      err_kws={"capsize": 3}, solid_capstyle="butt",
                      label=f"{OMEGA_UCB_1}"
                      )
    ax_14 = ax.twinx()
    sns.lineplot(x=data_rho_14[NORMALIZED_BUDGET].to_numpy(),
                 y=data_rho_14[REGRET].to_numpy(),
                 color=palette[1],
                 ax=ax_14,
                 estimator="mean",
                 err_style="bars",
                 errorbar=None,  # ("pi", 100),
                 err_kws={"capsize": 3}, solid_capstyle="butt",
                 label=f"{OMEGA_UCB_1_4}"
                 )
    ax.set_xscale("symlog", linthresh=.1)
    ax.legend()
    ax_14.legend()
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_14.get_legend_handles_labels()
    ax.get_legend().remove()
    ax_14.get_legend().remove()
    labels = labels1 + labels2
    handles = handles1 + handles2
    plt.gca().legend(handles, labels)
    # sns.move_legend(plt.gcf(), "lower left",
    #                 bbox_to_anchor=(.1, 9),
    #                 ncol=1, title=None, frameon=True)
    ax_14.set_xscale("symlog", linthresh=.1)
    ax.set_xlim((0.1, 1.05))
    ax_14.set_xlim((0.1, 1.05))
    ax.set_ylim((4700, 9500))
    ax_14.set_ylim((4000, 6500))
    ax.set_ylabel(r"Regret for $\rho=1$")
    ax_14.set_ylabel(r"Regret for $\rho=1/4$")
    ax.set_xlabel("Normalized Budget")

    plt.gcf().set_size_inches(cm2inch(16, 9))

    plt.tight_layout(pad=.5)
    # plt.subplots_adjust(top=.9)
    # plt.suptitle(r"$\omega$-UCB with $\rho=1$ and $\rho=1/4$ on Bernoulli bandit with $K=100$")
    # plt.title(r"Error bars show maximum and minimum")
    if with_ci:
        plt.savefig(os.path.join(os.getcwd(), "..", "figures", filename + "_ci" + "_curvature" + ".pdf"))
    else:
        plt.savefig(os.path.join(os.getcwd(), "..", "figures", filename + "_curvature" + ".pdf"))
    plt.show()


if __name__ == '__main__':
    filenames = [
        "synth_bernoulli",
    ]
    for filename in filenames:
        df = load_df(filename)
        df = prepare_df(df, n_steps=10)
        df[APPROACH][df[APPROACH] == "B-UCB"] = BUDGET_UCB
        df = df.loc[df[APPROACH] != OMEGA_UCB_1_64]
        df = df.loc[df[APPROACH] != OMEGA_UCB_1_32]
        df = df.loc[df[APPROACH] != OMEGA_UCB_1_16]
        df = df.loc[df[APPROACH] != OMEGA_UCB_1_8]
        # df = df.loc[df[APPROACH] != OMEGA_UCB_1_4]
        df = df.loc[df[APPROACH] != OMEGA_UCB_1_2]
        # df = df.loc[df[APPROACH] != OMEGA_UCB_1]
        df = df.loc[df[APPROACH] != OMEGA_UCB_2]
        df = df.loc[df[APPROACH] != OMEGA_STAR_UCB_1_64]
        df = df.loc[df[APPROACH] != OMEGA_STAR_UCB_1_32]
        df = df.loc[df[APPROACH] != OMEGA_STAR_UCB_1_16]
        df = df.loc[df[APPROACH] != OMEGA_STAR_UCB_1_8]
        df = df.loc[df[APPROACH] != OMEGA_STAR_UCB_1_4]
        df = df.loc[df[APPROACH] != OMEGA_STAR_UCB_1_2]
        df = df.loc[df[APPROACH] != OMEGA_STAR_UCB_1]
        df = df.loc[df[APPROACH] != OMEGA_STAR_UCB_2]
        df = df.loc[df[APPROACH] != UCB_SC_PLUS]
        df = df.loc[df[APPROACH] != BUDGET_UCB]
        df = df.loc[df[APPROACH] != BTS]
        df = df.loc[df[APPROACH] != B_GREEDY]
        df = df.loc[df[APPROACH] != CUCB]
        df = df.loc[df[APPROACH] != MUCB]
        df = df.loc[df[APPROACH] != IUCB]

        # plot_regret(df, filename, with_ci=True)
        plot_regret(df, filename, with_ci=False)
