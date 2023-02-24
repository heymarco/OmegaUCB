import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from util import load_df, prepare_df, cm2inch, create_palette, move_legend_below_graph
from components.bandit_logging import *
from approach_names import *
from colors import omega_ucb_base_color, eta_ucb_base_color, other_colors


import matplotlib as mpl
from matplotlib.legend import Legend
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
mpl.rc('font', family='serif')


def get_bts_hlines(df: pd.DataFrame):
    bts = df[df[APPROACH] == BTS].sort_values(by=K)
    upper = bts.groupby(K)[NORMALIZED_REGRET].quantile(q=0.75).reset_index().drop(K, axis=1)
    lower = bts.groupby(K)[NORMALIZED_REGRET].quantile(q=0.25).reset_index().drop(K, axis=1)
    median = bts.groupby(K)[NORMALIZED_REGRET].quantile().reset_index().drop(K, axis=1)
    return lower.to_numpy(), median.to_numpy(), upper.to_numpy()


def extend_legend(grid: sns.FacetGrid, line, text):
    legend = grid.legend
    title = legend.get_title().get_text()
    lines = legend.get_patches()
    lines.append(line)
    texts = legend.get_texts()
    texts = [t.get_text() for t in texts]
    texts.append(text)
    legend_data = {text: line for text, line in zip(texts, lines)}
    legend.remove()
    grid.add_legend(legend_data, title)



def plot_regret(df: pd.DataFrame):
    df = df[df[NORMALIZED_BUDGET] == 1]
    bts_quantiles = get_bts_hlines(df)
    df = df[df[APPROACH] != BTS]
    omega_ucb_color = sns.color_palette(omega_ucb_base_color, n_colors=4)[-2]
    eta_ucb_color = sns.color_palette(eta_ucb_base_color, n_colors=4)[-2]
    bts_color = sns.color_palette(other_colors, n_colors=4)[-2]
    palette = [omega_ucb_color, eta_ucb_color, bts_color]
    df[APPROACH] = df[APPROACH].apply(lambda x: ETA_UCB_ if ETA_UCB_ in x else x)
    df[APPROACH] = df[APPROACH].apply(lambda x: OMEGA_UCB_ if OMEGA_UCB_ in x else x)
    x = RHO
    y = NORMALIZED_REGRET
    hue = APPROACH
    col = K
    df = df.sort_values(by=[APPROACH, RHO])
    g = sns.catplot(data=df, kind="box", x=x, y=y, hue=hue, col=col, palette=palette[:2],
                    sharey=False, linewidth=1, showfliers=False)
    g.set(yscale="log")
    # lims = [0.00008, 0.02, 0.05]
    bts_line = None
    xtick_labels = [r"$\frac{1}{32}$", r"$\frac{1}{16}$", r"$\frac{1}{8}$",
                    r"$\frac{1}{4}$", r"$\frac{1}{2}$", "$1$", "$2$"]
    for i, ax in enumerate(g.axes.flatten()):
        # ax.set_ylim(bottom=lims[0])
        ax.axhspan(bts_quantiles[0][i], bts_quantiles[2][i], color=palette[-1], alpha=0.25, zorder=0, lw=0)
        bts_line = ax.axhline(bts_quantiles[1][i], color=palette[-1], zorder=0, label=BTS)
        ax.set_xticklabels(xtick_labels)
    extend_legend(g, bts_line, BTS)
    plt.gcf().set_size_inches(cm2inch(20, 5))
    plt.tight_layout(pad=.5)
    plt.subplots_adjust(right=.88)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", "omega_vs_eta.pdf"))
    plt.show()


if __name__ == '__main__':
    filename = "synth_beta"
    df = load_df(filename)
    df = prepare_df(df, n_steps=50)
    # df = df.loc[df[APPROACH] != OMEGA_UCB_1_5]
    # df = df.loc[df[APPROACH] != OMEGA_UCB_1_6]
    # df = df.loc[df[APPROACH] != OMEGA_UCB_1_4]
    # df = df.loc[df[APPROACH] != OMEGA_UCB_1_3]
    # df = df.loc[df[APPROACH] != OMEGA_UCB_1_2]
    # df = df.loc[df[APPROACH] != OMEGA_UCB_1]
    # df = df.loc[df[APPROACH] != OMEGA_UCB_2]
    # df = df.loc[df[APPROACH] != OMEGA_UCB_3]
    # df = df.loc[df[APPROACH] != OMEGA_UCB_4]
    # df = df.loc[df[APPROACH] != ETA_UCB_1_5]
    # df = df.loc[df[APPROACH] != ETA_UCB__6]
    # df = df.loc[df[APPROACH] != ETA_UCB_1_4]
    # df = df.loc[df[APPROACH] != ETA_UCB_1_3]
    # df = df.loc[df[APPROACH] != ETA_UCB_1_2]
    # df = df.loc[df[APPROACH] != ETA_UCB_1]
    # df = df.loc[df[APPROACH] != ETA_UCB_2]
    # df = df.loc[df[APPROACH] != ETA_UCB_3]
    # df = df.loc[df[APPROACH] != ETA_UCB_4]
    df = df.loc[df[APPROACH] != UCB_SC_PLUS]
    df = df.loc[df[APPROACH] != B_GREEDY]
    # df = df.loc[df[APPROACH] != BTS]
    df = df.loc[df[APPROACH] != CUCB]
    df = df.loc[df[APPROACH] != MUCB]
    df = df.loc[df[APPROACH] != IUCB]
    df = df.loc[df[APPROACH] != BUDGET_UCB]
    plot_regret(df)