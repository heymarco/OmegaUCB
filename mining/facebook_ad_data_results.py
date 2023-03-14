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
from scipy.stats import gmean

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
mpl.rc('font', family='serif')


def compute_ylims(df: pd.DataFrame, x, hue, x_cut=.7):
    lims = []
    df = df.groupby([x, hue]).mean().reset_index()
    df = df[df[x] <= x_cut]
    max_regret = df[NORMALIZED_REGRET].max()
    min_regret = df[NORMALIZED_REGRET].min()
    lims.append((min_regret * 0.8, max_regret))
    return lims


def plot_regret(df: pd.DataFrame, filename: str, x_cut: float):
    # df = df[df[K] == 10]
    df = df.sort_values(by=APPROACH)
    x = NORMALIZED_BUDGET
    y = NORMALIZED_REGRET
    hue = APPROACH
    lims = compute_ylims(df, x, hue, x_cut=x_cut)
    palette = create_palette(df)
    markers = get_markers_for_approaches(np.unique(df[APPROACH]))
    g = sns.relplot(data=df, x=x, y=y, hue=hue, lw=1,
                    # markersize=3,
                    markeredgewidth=0.1,
                    kind="line", palette=palette, legend=False, errorbar=None,
                    facet_kws={"sharey": False}, style=hue, markers=markers, dashes=False)
    g.set(xscale="log")
    for lim, ax in zip(lims, g.axes.flatten()):
        ax.set_ylim(lim)
    plt.gcf().set_size_inches(cm2inch((20 / 2, 7.5 * 0.55)))
    plt.tight_layout(pad=.5)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", filename + ".pdf"))
    plt.show()


if __name__ == '__main__':
    filenames = [
        "facebook_beta_combined",
        "facebook_bernoulli"
    ]
    x_cuts = [1.0, 0.3]
    for cut, filename in zip(x_cuts, filenames):
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
            # df = df.loc[df[APPROACH] != OMEGA_UCB_2]
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
            df = df.loc[df[APPROACH] != ETA_UCB_1_64]
            df = df.loc[df[APPROACH] != ETA_UCB_1_32]
            df = df.loc[df[APPROACH] != ETA_UCB_1_16]
            df = df.loc[df[APPROACH] != ETA_UCB_1_8]
            df = df.loc[df[APPROACH] != ETA_UCB_1_4]
            df = df.loc[df[APPROACH] != ETA_UCB_1_2]
            df = df.loc[df[APPROACH] != ETA_UCB_1]
            df = df.loc[df[APPROACH] != ETA_UCB_2]
            # df = df.loc[df[APPROACH] != UCB_SC_PLUS]
            # df = df.loc[df[APPROACH] != BUDGET_UCB]
        # df = df.loc[df[APPROACH] != IUCB]
        # df = df.loc[df[APPROACH] != MUCB]
        # df = df.loc[df[APPROACH] != CUCB]
        # df = df.loc[df[APPROACH] != BTS]
        # df = df.loc[df[APPROACH] != B_GREEDY]
        plot_regret(df, filename, x_cut=cut)
