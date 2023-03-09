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
    bts = df[df[APPROACH] == BTS]
    upper = bts.groupby(["Distribution", K])[NORMALIZED_REGRET].quantile(q=0.75).reset_index().drop([K, "Distribution"], axis=1)
    lower = bts.groupby(["Distribution", K])[NORMALIZED_REGRET].quantile(q=0.25).reset_index().drop([K, "Distribution"], axis=1)
    median = bts.groupby(["Distribution", K])[NORMALIZED_REGRET].quantile().reset_index().drop([K, "Distribution"], axis=1)
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


def plot_regret(df: pd.DataFrame, figsize, figname):
    df = df[df[NORMALIZED_BUDGET] == 1]
    df = df[df[APPROACH] != BTS]
    df.sort_values(by=["Distribution", K])
    omega_ucb_color = sns.color_palette(omega_ucb_base_color, n_colors=12)[3]
    eta_ucb_color = sns.color_palette(eta_ucb_base_color, n_colors=12)[2]
    palette = [omega_ucb_color, eta_ucb_color]
    df[APPROACH] = df[APPROACH].apply(lambda x: ETA_UCB_ if ETA_UCB_ in x else x)
    df[APPROACH] = df[APPROACH].apply(lambda x: OMEGA_UCB_ if OMEGA_UCB_ in x else x)
    x = RHO
    y = NORMALIZED_REGRET
    hue = APPROACH
    col = K
    row = "Distribution"
    g = sns.catplot(data=df, kind="bar", x=x, y=y, hue=hue, col=col, row=row, palette=palette,
                    sharey=False, sharex=True, linewidth=.8, errwidth=1)
    g.set(yscale="log")
    # lims = [0.00008, 0.02, 0.05]
    bts_line = None
    xtick_labels = [r"$\frac{1}{64}$", r"$\frac{1}{32}$", r"$\frac{1}{16}$", r"$\frac{1}{8}$",
                    r"$\frac{1}{4}$", r"$\frac{1}{2}$", "$1$", "$2$"
                    ]
    for i, ax in enumerate(g.axes.flatten()):
        # ax.set_ylim(bottom=lims[0])
        # ax.axhspan(bts_quantiles[0][i], bts_quantiles[2][i], color=palette[-1], alpha=0.25, zorder=0, lw=0)
        # bts_line = ax.axhline(bts_quantiles[1][i], color=palette[-1], zorder=0, label=BTS)
        ax.set_xticklabels(xtick_labels)
        ax_title = ax.get_title()
        dist_title, K_title = ax_title.split(" | ")
        dist_title = dist_title.split(" = ")[-1]
        if i < 3:
            ax.set_title(K_title)
        else:
            ax.set_title("")
        if i % 3 == 0:
            ax.set_ylabel(r"Regret ({})".format(dist_title))
    extend_legend(g, bts_line, BTS)
    plt.gcf().set_size_inches(cm2inch(figsize))
    plt.tight_layout(pad=.7)
    plt.subplots_adjust(right=0.86, wspace=.4, hspace=.13)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", figname + ".pdf"))
    plt.show()

    df = df.groupby([row, col, hue]).mean().reset_index()
    print(df)


if __name__ == '__main__':
    filename = "synth_beta"
    df_beta = load_df(filename)
    df_beta = prepare_df(df_beta, n_steps=10)
    df_beta = df_beta.loc[df_beta[APPROACH] != UCB_SC_PLUS]
    df_beta = df_beta.loc[df_beta[APPROACH] != B_GREEDY]
    df_beta = df_beta.loc[df_beta[APPROACH] != CUCB]
    df_beta = df_beta.loc[df_beta[APPROACH] != MUCB]
    df_beta = df_beta.loc[df_beta[APPROACH] != IUCB]
    df_beta = df_beta.loc[df_beta[APPROACH] != BUDGET_UCB]
    # df_beta = df_beta.loc[df_beta[APPROACH] != OMEGA_UCB_2]
    # df_beta = df_beta.loc[df_beta[APPROACH] != ETA_UCB_2]
    # plot_regret(df_beta, figsize=(20 * 1.8 / 3, 5), figname="sensitivity_" + filename, legend=True, subplots_adjust_right=0.78)
    df_beta["Distribution"] = "Beta"

    filename = "synth_bernoulli"
    df_bern = load_df(filename)
    df_bern = prepare_df(df_bern, n_steps=10)
    df_bern = df_bern.loc[df_bern[APPROACH] != UCB_SC_PLUS]
    df_bern = df_bern.loc[df_bern[APPROACH] != B_GREEDY]
    df_bern = df_bern.loc[df_bern[APPROACH] != CUCB]
    df_bern = df_bern.loc[df_bern[APPROACH] != MUCB]
    df_bern = df_bern.loc[df_bern[APPROACH] != IUCB]
    df_bern = df_bern.loc[df_bern[APPROACH] != BUDGET_UCB]
    # df_bern = df_bern.loc[df_bern[APPROACH] != OMEGA_UCB_2]
    # df_bern = df_bern.loc[df_bern[APPROACH] != ETA_UCB_2]

    df_bern["Distribution"] = "Bernoulli"
    df = pd.concat([df_bern, df_beta]).reset_index()

    plot_regret(df, figsize=(20, 7), figname="sensitivity_study")
