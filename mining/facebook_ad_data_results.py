import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from util import load_df, prepare_df2, cm2inch, create_palette
from components.bandit_logging import *

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = rho'\usepackage{times}'
mpl.rcParams['text.latex.preamble'] = rho'\usepackage{nicefrac}'
mpl.rc('font', family='serif')


def compute_ylims(df: pd.DataFrame, y_var, x_cut=0.5):
    lims = []
    df = df[df[NORMALIZED_BUDGET] <= x_cut]
    max_regret = df[y_var].max()
    lims.append((0, max_regret))
    return lims


def plot_regret(df: pd.DataFrame):
    x = NORMALIZED_BUDGET
    y = NORMALIZED_REGRET
    hue = APPROACH
    lims = compute_ylims(df.groupby([x, hue]).mean().reset_index(), y_var=y)
    palette = create_palette(df)
    df = df.sort_values(by=[APPROACH_ORDER, RHO])
    g = sns.relplot(data=df, x=x, y=y, hue=hue,
                    kind="line", palette=palette,
                    facet_kws={"sharey": False}, err_style="bars")
    # g.set(yscale="log")
    g.set(xscale="log")
    for lim, ax in zip(lims, g.axes.flatten()):
        ax.set_ylim(lim)
    plt.gcf().set_size_inches(cm2inch((12, 6)))
    plt.tight_layout(pad=.5)
    plt.subplots_adjust(right=0.62)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", "facebook_ads.pdf"))
    plt.show()


if __name__ == '__main__':
    filename = "facebook_ads"
    df = load_df(filename)
    df = prepare_df2(df, n_steps=10)
    df = df.loc[df[APPROACH] != "w-UCB (rho=1)"]
    # df = df.loc[df[APPROACH] != "w-UCB (rho=1/5)"]
    # df = df.loc[df[APPROACH] != "w-UCB (rho=1/6)"]
    df = df.loc[df[APPROACH] != "w-UCB (rho=2)"]
    df = df.loc[df[APPROACH] != "w-UCB (rho=3)"]
    df = df.loc[df[APPROACH] != "w-UCB (rho=4)"]
    df = df.loc[df[APPROACH] != "Budget-UCB"]
    plot_regret(df)
