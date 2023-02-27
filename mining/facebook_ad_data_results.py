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
from scipy.stats import gmean

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
mpl.rc('font', family='serif')


def compute_ylims(df: pd.DataFrame, x, hue, x_cut=0.3):
    lims = []
    df = df.groupby([x, hue]).mean().reset_index()
    df = df[df[x] <= x_cut]
    max_regret = df[NORMALIZED_REGRET].max()
    min_regret = df[NORMALIZED_REGRET].min()
    lims.append((min_regret * 0.8, max_regret))
    return lims


def plot_regret(df: pd.DataFrame, filename: str):
    # df = df[df[K] == 57]
    x = NORMALIZED_BUDGET
    y = NORMALIZED_REGRET
    hue = APPROACH
    lims = compute_ylims(df, x, hue)
    palette = create_palette(df)
    g = sns.relplot(data=df, x=x, y=y, hue=hue,
                    kind="line", palette=palette, legend=False,
                    facet_kws={"sharey": False}, errorbar=("se", 1),
                    err_style="bars")
    g.set(xscale="log")
    for lim, ax in zip(lims, g.axes.flatten()):
        ax.set_ylim(lim)
    plt.gcf().set_size_inches(cm2inch((20 / 3, 7.5 * 0.55)))
    plt.tight_layout(pad=.5)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", filename + ".pdf"))
    plt.show()


if __name__ == '__main__':
    filenames = [
        "facebook_beta",
        "facebook_bernoulli"
    ]
    for filename in filenames:
        df = load_df(filename)
        df = prepare_df(df, n_steps=10)
        # df = df[np.logical_and(df[K] > 20, df[K] < 60)]
        if "beta" in filename:
            df = df.loc[df[APPROACH] != OMEGA_UCB_1_64]
            df = df.loc[df[APPROACH] != OMEGA_UCB_1_32]
            df = df.loc[df[APPROACH] != OMEGA_UCB_1_16]
            df = df.loc[df[APPROACH] != OMEGA_UCB_1_8]
            df = df.loc[df[APPROACH] != OMEGA_UCB_1_4]
            df = df.loc[df[APPROACH] != OMEGA_UCB_1_2]
            df = df.loc[df[APPROACH] != OMEGA_UCB_1]
            df = df.loc[df[APPROACH] != OMEGA_UCB_2]
            df = df.loc[df[APPROACH] != ETA_UCB_1_64]
            df = df.loc[df[APPROACH] != ETA_UCB_1_32]
            df = df.loc[df[APPROACH] != ETA_UCB_1_16]
            df = df.loc[df[APPROACH] != ETA_UCB_1_8]
            df = df.loc[df[APPROACH] != ETA_UCB_1_4]
            df = df.loc[df[APPROACH] != ETA_UCB_1_2]
            # df = df.loc[df[APPROACH] != ETA_UCB_1]
            df = df.loc[df[APPROACH] != ETA_UCB_2]
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
        df = df.loc[df[APPROACH] != UCB_SC_PLUS]
        df = df.loc[df[APPROACH] != BUDGET_UCB]
        # df = df.loc[df[APPROACH] != IUCB]
        # df = df.loc[df[APPROACH] != MUCB]
        # df = df.loc[df[APPROACH] != CUCB]
        # df = df.loc[df[APPROACH] != BTS]
        # df = df.loc[df[APPROACH] != B_GREEDY]
        print(df)
        plot_regret(df, filename)
