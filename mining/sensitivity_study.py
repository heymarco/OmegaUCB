import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from util import load_df, prepare_df, cm2inch, create_palette
from components.bandit_logging import *


def plot_sensitivity_study(df: pd.DataFrame):
    x = RHO
    y = REGRET
    hue = MINIMUM_AVERAGE_COST
    col = K
    style = hue
    palette = create_palette(df)
    g = sns.relplot(data=df, x=x, y=y, hue=hue, col=col, style=style, markers=True,
                    kind="line", palette=palette, aspect=1, height=cm2inch(4)[0],
                    facet_kws={"sharey": False}, err_style=None)
    g.set(yscale="log")
    # g.set(xscale="log")
    plt.show()


if __name__ == '__main__':
    filename = "synth_beta"
    df = load_df(filename)
    df = prepare_df(df)
    df = df.loc[df[IS_OUR_APPROACH]]
    plot_sensitivity_study(df)


