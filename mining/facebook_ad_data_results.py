import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from util import load_df, prepare_df2, cm2inch, create_palette
from components.bandit_logging import *
from approach_names import *

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{times}'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{nicefrac}'
mpl.rc('font', family='serif')


def compute_ylims(df: pd.DataFrame, y_var, x_cut=0.3):
    lims = []
    df = df[df[NORMALIZED_BUDGET] <= x_cut]
    max_regret = df[y_var].max()
    lims.append((0, max_regret))
    return lims


def plot_regret(df: pd.DataFrame, filename: str):
    x = NORMALIZED_BUDGET
    y = NORMALIZED_REGRET
    hue = APPROACH
    lims = compute_ylims(df.groupby([x, hue]).mean().reset_index(), y_var=y)
    palette = create_palette(df)
    df = df.sort_values(by=[APPROACH_ORDER, RHO])
    g = sns.relplot(data=df, x=x, y=y, hue=hue,
                    kind="line", palette=palette,
                    facet_kws={"sharey": False}, err_style="bars")
    g.set(xscale="log")
    for lim, ax in zip(lims, g.axes.flatten()):
        ax.set_ylim(lim)
    plt.gcf().set_size_inches(cm2inch((12, 6)))
    plt.tight_layout(pad=.5)
    plt.subplots_adjust(right=0.62)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", filename + ".pdf"))
    plt.show()


if __name__ == '__main__':
    filenames = ["facebook_beta", "facebook_bernoulli"]
    for filename in filenames:
        df = load_df(filename)
        df = prepare_df2(df, n_steps=10)
        # df = df.loc[df[APPROACH] != OMEGA_UCB_1_5]
        # df = df.loc[df[APPROACH] != OMEGA_UCB_1_6]
        # df = df.loc[df[APPROACH] != OMEGA_UCB_1_4]
        # df = df.loc[df[APPROACH] != OMEGA_UCB_1_3]
        # df = df.loc[df[APPROACH] != OMEGA_UCB_1_2]
        df = df.loc[df[APPROACH] != OMEGA_UCB_1]
        df = df.loc[df[APPROACH] != OMEGA_UCB_2]
        df = df.loc[df[APPROACH] != OMEGA_UCB_3]
        df = df.loc[df[APPROACH] != OMEGA_UCB_4]
        df = df.loc[df[APPROACH] != ETA_UCB_1_5]
        df = df.loc[df[APPROACH] != ETA_UCB_1_6]
        df = df.loc[df[APPROACH] != ETA_UCB_1_4]
        df = df.loc[df[APPROACH] != ETA_UCB_1_3]
        df = df.loc[df[APPROACH] != ETA_UCB_1_2]
        df = df.loc[df[APPROACH] != ETA_UCB_1]
        df = df.loc[df[APPROACH] != ETA_UCB_2]
        df = df.loc[df[APPROACH] != ETA_UCB_3]
        df = df.loc[df[APPROACH] != ETA_UCB_4]
        plot_regret(df, filename)
