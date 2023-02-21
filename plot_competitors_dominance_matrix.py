import copy
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from util import cm2inch

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
mpl.rc('font', family='serif')


if __name__ == '__main__':
    approaches = [
        r"$\epsilon$-first",
        "KUBE",
        "UCB-BV1",
        "PD-BwK",
        "Budget-UCB",
        "BTS",
        "MRCB",
        "m-UCB",
        "b-greedy",
        "c-UCB",
        "i-UCB",
        "KL-UCB-SC",
        "UCB-SC+",
        r"$\omega$-UCB"
    ]

    approaches_with_years = [
        r"$\epsilon$-first (2010)",
        "KUBE (2012)",
        "UCB-BV1 (2013)",
        "PD-BwK (2013)",
        "Budget-UCB (2015)",
        "BTS (2015)",
        "MRCB (2016)",
        "m-UCB (2017)",
        "b-greedy (2017)",
        "c-UCB (2017)",
        "i-UCB (2017)",
        "KL-UCB-SC (2017)",
        "UCB-SC+ (2018)",
        r"$\omega$-UCB (ours)"
    ]

    old_matrix = [
        [0, -1, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, -2],
        [1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, -2],
        [-2, 1, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, -2],
        [-2, -2, -2, 0, 0, -1, -1, -1, -1, -1, 0, -1, -2],
        [-2, -2, 1, -2, 0, 0, 0, 0, 0, 0, 0, 0, -1],
        [1, 1, 1, 1, -2, 0, -1, -1, -1, -1, 0, 0, -1],
        [1, -2, -2, 1, -2, 1, 0, 0, 0, 0, 0, 0, -2],
        [1, -2, -2, 1, -2, 1, 0, 0, -1, -1, 0, 0, -1],
        [1, -2, -2, 1, -2, 1, 0, 1, 0, 0, 0, 0, -1],
        [1, -2, -2, 1, -2, 1, 0, 1, 0, 0, 0, 0, -1],
        [-2, 0, 1, 0, -2, 0, -2, -2, -2, -2, 0, 0, 0],
        [-2, -2, 1, 1, -2, 0, -2, -2, -2, -2, 0, 0, -1],
        [-2, -2, -2, -2, 1, 1, -2, 1, 1, 1, 0, 1, 0]
    ]

    matrix = [
        [0, -1, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -2],
        [1, 0, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -2],
        [-2, 1, 0, 0, -1, -1, -2, 0, 0, 0, 0, -1, -1, -2],
        [-2, -2, -2, 0, 0, -1, -2, -1, -1, -1, -1, 0, -1, -2],
        [-2, -2, 1, -2, 0, 0, -2, 0, 0, 0, 0, 0, 0, -1],
        [1, 1, 1, 1, -2, 0, -1, -1, -1, -1, -1, 0, 0, -1],
        [1, 1, -2, -2, -2, 1, 0, -1, -2, -1, -1, -2, -2, -1],
        [1, -2, -2, 1, -2, 1, 1, 0, 1, -1, -1, 0, 0, -1],
        [1, -2, -2, 1, -2, 1, -2, -1, 0, -1, -1, 0, 0, -1],
        [1, -2, -2, 1, -2, 1, 1, 1, 1, 0, 0, 0, 0, -1],
        [1, -2, -2, 1, -2, 1, 1, 1, 1, 0, 0, 0, 0, -1],
        [-2, 0, 1, 0, -2, 0, -2, -2, -2, -2, -2, 0, -2, -2],
        [-2, -2, 1, 1, -2, 0, -2, -2, -2, -2, -2, -2, 0, -1],
        [-2, -2, -2, -2, 1, 1, 1, 1, 1, 1, 1, -2, 1, 0]
    ]

    matrix = np.array(matrix)

    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            if matrix[row][col] == -2:
                matrix[col][row] = -2

    matrix_inv = [matrix[-i-1] for i in range(len(matrix))]
    mask = np.triu(np.ones_like(matrix_inv))
    mask = np.array([mask[-(i+1)] for i in range(len(mask))])
    approaches_inv = [approaches_with_years[-(i+1)] for i in range(len(approaches_with_years))]

    first_row = matrix[7, :].copy()
    matrix[7, :] = matrix[8, :]
    matrix[8, :] = first_row

    col_7 = matrix[:, 7].copy()
    matrix[:, 7] = matrix[:, 8]
    matrix[:, 8] = col_7

    show_full = True
    palette = list(sns.color_palette("vlag_r", n_colors=101))
    palette = [palette[27]] + [palette[50]] + [palette[87]]
    palette = palette if show_full else palette[1:]
    palette = ["lightgray"] + palette


    if not show_full:
        df = pd.DataFrame(np.array(matrix_inv), columns=approaches, index=approaches_inv)
        ax = sns.heatmap(df.iloc[:-1, :-1], cmap=palette,
                         linewidth=1, cbar=False, square=True, mask=mask[:-1, :-1])
    else:
        df = pd.DataFrame(np.array(matrix), columns=approaches, index=approaches_with_years)
        ax = sns.heatmap(df, cmap=palette,
                         linewidth=1, cbar=False, square=True)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.set_label_position('top')
    ax.yaxis.set_label_position('left')
    ax.xaxis.tick_top()
    # ax.set_title("Dominance matrix of related approaches")
    for item in ax.get_xticklabels():
        item.set_rotation(90)
    for item in ax.get_yticklabels():
        item.set_rotation(0)
    plt.gcf().set_size_inches(cm2inch((12, 9)))
    plt.tight_layout(pad=.5)
    plt.savefig(os.path.join(os.getcwd(), "figures", "dominance_table.pdf"))
    plt.show()