import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from util import cm2inch

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{times}'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{nicefrac}'
mpl.rc('font', family='serif')


if __name__ == '__main__':
    approaches = [
        r"$\epsilon$-first",
        "KUBE",
        "UCB-BV1",
        "PD-BwK",
        "Budget-UCB",
        "BTS",
        "b-greedy",
        "m-UCB",
        "c-UCB",
        "i-UCB",
        "KL-UCB-SC",
        "UCB-SC+",
    ]

    matrix = [
        [0, -1, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0],
        [1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1],
        [0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0],
        [1, 0, 0, 1, 0, 1, 1, 0, -1, -1, 0, 0],
        [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    matrix_bin = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    use_bin = True
    if use_bin:
        df = pd.DataFrame(np.array(matrix_bin).T, columns=approaches, index=approaches)
        ax = sns.heatmap(df.iloc[:-1, 1:], cmap=sns.color_palette("vlag_r", n_colors=3)[1:],
                         linewidth=1, cbar=False, square=True, mask=np.tril(np.ones_like(matrix_bin))[:-1, 1:])
    else:
        df = pd.DataFrame(np.array(matrix).T, columns=approaches, index=approaches)
        ax = sns.heatmap(df, cmap=sns.color_palette("vlag_r", n_colors=3),
                         linewidth=1, cbar=False, square=True, mask=np.tril(np.ones_like(matrix_bin)))
    ax.set_ylabel("Competitor")
    ax.set_xlabel("Approach outperforms")
    ax.xaxis.set_label_position('top')
    ax.yaxis.set_label_position('right')
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    # ax.set_title("Dominance matrix of related approaches")
    for item in ax.get_xticklabels():
        item.set_rotation(90)
    for item in ax.get_yticklabels():
        item.set_rotation(0)
    plt.gcf().set_size_inches(cm2inch((12, 9)))
    plt.tight_layout(pad=.5)
    plt.savefig(os.path.join(os.getcwd(), "figures", "dominance_table.pdf"))
    plt.show()