import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from util import load_df, prepare_df, cm2inch
from components.bandit_logging import *
from approach_names import *
from colors import omega_ucb_base_color, eta_ucb_base_color

import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
mpl.rc('font', family='serif')


def plot_regret(df: pd.DataFrame, figsize, figname, narrow):
    df = df[df[NORMALIZED_BUDGET] == 1.0]
    df = df[df[APPROACH] != BTS]
    if narrow:
        df = df[df[K] == 50]
    df.sort_values(by=["Distribution", K])
    omega_ucb_color = sns.color_palette(omega_ucb_base_color, n_colors=12)[3]
    eta_ucb_color = sns.color_palette(eta_ucb_base_color, n_colors=12)[2]
    palette = [omega_ucb_color, eta_ucb_color]
    df[APPROACH] = df[APPROACH].apply(lambda x: OMEGA_STAR_UCB_ if OMEGA_STAR_UCB_ in x else x)
    df[APPROACH] = df[APPROACH].apply(lambda x: OMEGA_UCB_ if OMEGA_UCB_ in x else x)
    x = RHO
    y = REGRET
    hue = APPROACH
    col = K
    row = "Distribution"
    if narrow:
        g = sns.catplot(data=df, kind="bar",
                        x=x, y=y, hue=hue, col=row,
                        palette=palette,
                        sharey=False, sharex=True,
                        linewidth=.8, errwidth=1)
    else:
        g = sns.catplot(data=df, kind="bar",
                        x=x, y=y, hue=hue, col=col, row=row,
                        palette=palette,
                        sharey=False, sharex=True,
                        linewidth=.8, errwidth=1)
    g.set(yscale="log")
    if narrow:
        sns.move_legend(g, "center right", title="", ncol=1, frameon=True)
    else:
        sns.move_legend(g, "upper center", title="", ncol=2, frameon=True)
    xtick_labels = [r"$\frac{1}{64}$", r"$\frac{1}{32}$", r"$\frac{1}{16}$", r"$\frac{1}{8}$",
                    r"$\frac{1}{4}$", r"$\frac{1}{2}$", "$1$", "$2$"]
    for i, ax in enumerate(g.axes.flatten()):
        ax.set_xticklabels(xtick_labels)
        ax_title = ax.get_title()
        if narrow:
            dist_title = ax_title.split(" = ")[-1]
        else:
            dist_title, K_title = ax_title.split(" | ")
            dist_title = dist_title.split(" = ")[-1]
        if not narrow:
            if i <= 2:
                ax.set_title(K_title)
            else:
                ax.set_title("")
        else:
            ax.set_title("")
        if dist_title == "Bernoulli":
            id = "Br"
        elif dist_title == "Gen. Bern.":
            id = "GBr"
        elif dist_title == "Beta":
            id = "Bt"
        else:
            raise ValueError
        if narrow:
            ax.set_ylabel(r"Regret ({})".format(id))
        else:
            if i % 2 == 0:
                ax.set_ylabel(r"Regret ({})".format(id))
        if i % 3 > 0:
            ax.set_ylabel("")
    plt.gcf().set_size_inches(cm2inch(figsize))
    plt.tight_layout(pad=.7)
    if narrow:
        plt.subplots_adjust(right=.85)
        plt.savefig(os.path.join(os.getcwd(), "..", "figures", figname + "_narrow" + ".pdf"))
    else:
        plt.subplots_adjust(wspace=.4, hspace=.13, top=0.8)
        plt.savefig(os.path.join(os.getcwd(), "..", "figures", figname + ".pdf"))
    plt.show()


if __name__ == '__main__':
    narrow = False
    filename = "synth_beta"
    df_beta = load_df(filename)
    df_beta = prepare_df(df_beta, n_steps=10)
    df_beta = df_beta.loc[df_beta[APPROACH] != UCB_SC_PLUS]
    df_beta = df_beta.loc[df_beta[APPROACH] != B_GREEDY]
    df_beta = df_beta.loc[df_beta[APPROACH] != CUCB]
    df_beta = df_beta.loc[df_beta[APPROACH] != MUCB]
    df_beta = df_beta.loc[df_beta[APPROACH] != IUCB]
    df_beta = df_beta.loc[df_beta[APPROACH] != BUDGET_UCB]
    df_beta = df_beta.loc[df_beta[APPROACH] != "B-UCB"]
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
    df_bern = df_bern.loc[df_bern[APPROACH] != "B-UCB"]
    df_bern["Distribution"] = "Bernoulli"

    filename = "synth_multinomial"
    df_mult = load_df(filename)
    df_mult = prepare_df(df_mult, n_steps=10)
    df_mult = df_mult.loc[df_mult[APPROACH] != UCB_SC_PLUS]
    df_mult = df_mult.loc[df_mult[APPROACH] != B_GREEDY]
    df_mult = df_mult.loc[df_mult[APPROACH] != CUCB]
    df_mult = df_mult.loc[df_mult[APPROACH] != MUCB]
    df_mult = df_mult.loc[df_mult[APPROACH] != IUCB]
    df_mult = df_mult.loc[df_mult[APPROACH] != BUDGET_UCB]
    df_mult = df_mult.loc[df_mult[APPROACH] != "B-UCB"]
    df_mult["Distribution"] = "Gen. Bern."

    df = pd.concat([df_bern, df_mult, df_beta]).reset_index(drop=True)

    if narrow:
        plot_regret(df, figsize=(20, 6 * 0.75), figname="sensitivity_study", narrow=narrow)
    else:
        plot_regret(df, figsize=(18, 4.8 * 1.5), figname="sensitivity_study", narrow=narrow)
