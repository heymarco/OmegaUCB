import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from colors import get_markers_for_approaches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from util import load_df, prepare_df, cm2inch, create_palette
from components.bandit_logging import *
from approach_names import *

import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')


def plot_regret(df: pd.DataFrame, filename: str, with_ci: bool = False):
    df = df.iloc[::-1]
    df.loc[df[APPROACH] == "B-UCB", APPROACH] = BUDGET_UCB
    x = NORMALIZED_BUDGET
    y = REGRET
    hue = APPROACH
    lims = [(0, 1500), (0, 2000)]
    palette = create_palette(df)
    markers = get_markers_for_approaches(np.unique(df[APPROACH]))
    if with_ci:
        g = sns.relplot(data=df, x=x, y=y, hue=hue, lw=1,
                        markeredgewidth=0.1,
                        kind="line", palette=palette, legend=False,
                        errorbar="ci", err_style="bars", seed=0, n_boot=500, err_kws={"capsize": 2},
                        solid_capstyle="butt",
                        facet_kws={"sharey": False}, style=hue, markers=False, dashes=False)
    else:
        g = sns.relplot(data=df, x=x, y=y, hue=hue, lw=1,
                        markeredgewidth=0.1,
                        kind="line", palette=palette, legend=False,
                        errorbar=None,
                        facet_kws={"sharey": False}, style=hue, markers=markers, dashes=False)

    for i, (lim, ax) in enumerate(zip(lims, g.axes.flatten())):
        ax.set_ylim(lim)
        ax.set_xlim((0.095, 1))
        ax.set_xscale("symlog", linthresh=.1)
    plt.gcf().set_size_inches(cm2inch(18 / 2.8, 5.8 * 0.58))
    plt.tight_layout(pad=.5)
    if with_ci:
        plt.savefig(os.path.join(os.getcwd(), "..", "figures", filename + "_ci" + ".pdf"))
    else:
        plt.savefig(os.path.join(os.getcwd(), "..", "figures", filename + ".pdf"))
    plt.show()


if __name__ == '__main__':
    filenames = [
        "facebook_beta",
        "facebook_bernoulli"
    ]
    setting_ids = [
        "FB-Br",
        "FB-Bt"
    ]

    for filename in filenames:
        df = load_df(filename)
        df = prepare_df(df, n_steps=10)
        if "beta" in filename:
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
            # df = df.loc[df[APPROACH] != ETA_UCB_1_4]
            df = df.loc[df[APPROACH] != OMEGA_STAR_UCB_1_2]
            # df = df.loc[df[APPROACH] != ETA_UCB_1]
            df = df.loc[df[APPROACH] != OMEGA_STAR_UCB_2]
            # df = df.loc[df[APPROACH] != UCB_SC_PLUS]
            # df = df.loc[df[APPROACH] != BUDGET_UCB]
            df["setting"] = "FB-Bt"
            pass
        if "bernoulli" in filename:
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
            # df = df.loc[df[APPROACH] != UCB_SC_PLUS]
            # df = df.loc[df[APPROACH] != BUDGET_UCB]
            df["setting"] = "FB-Br"
        # df = df.loc[df[APPROACH] != IUCB]
        # df = df.loc[df[APPROACH] != MUCB]
        # df = df.loc[df[APPROACH] != CUCB]
        # df = df.loc[df[APPROACH] != BTS]
        # df = df.loc[df[APPROACH] != B_GREEDY]

        plot_regret(df, filename, with_ci=True)
        plot_regret(df, filename, with_ci=False)
