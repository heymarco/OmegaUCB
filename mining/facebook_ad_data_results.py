import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from util import load_df, prepare_df2, cm2inch, create_palette
from components.bandit_logging import *


def compute_ylims(df: pd.DataFrame, row_var, x_cut=0.5):
    lims = []
    df = df[df[NORMALIZED_BUDGET] <= x_cut]
    df.sort_values(by=[row_var], inplace=True)
    for _, row_df in df.groupby(row_var):
            max_regret = row_df[REGRET].max()
            lims.append((0, max_regret))
    return lims


def plot_regret(df: pd.DataFrame):
    x = NORMALIZED_BUDGET
    y = REGRET
    hue = APPROACH
    row = MINIMUM_AVERAGE_COST
    df = df.groupby([x, hue, row]).mean().reset_index()
    lims = compute_ylims(df, row_var=row)
    palette = create_palette(df)
    df = df.sort_values(by=APPROACH_ORDER)
    g = sns.relplot(data=df, x=x, y=y, hue=hue, row=row,
                    kind="line", palette=palette, aspect=1.8, height=cm2inch(6)[0],
                    facet_kws={"sharey": False}, err_style=None)
    # g.set(yscale="log")
    # g.set(xscale="log")
    for lim, ax in zip(lims, g.axes.flatten()):
        ax.set_ylim(lim)
    plt.show()


if __name__ == '__main__':
    filename = "facebook_ads"
    df = load_df(filename)
    df = prepare_df2(df)
    # df = df.loc[df[APPROACH] != "w-UCB (a, r=1)"]
    df = df.loc[df[APPROACH] != "w-UCB (rho=1/5)"]
    df = df.loc[df[APPROACH] != "w-UCB (rho=1/6)"]
    df = df.loc[df[APPROACH] != "w-UCB (rho=2)"]
    df = df.loc[df[APPROACH] != "w-UCB (rho=3)"]
    df = df.loc[df[APPROACH] != "w-UCB (rho=4)"]
    # df = df.loc[df[APPROACH] != "UCB-SC+"]
    plot_regret(df)
