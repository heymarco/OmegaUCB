import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from util import cm2inch

import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')

if __name__ == '__main__':
    # This code will reproduce figure 2 in the paper

    # Define the names of the approaches
    approaches = [
        r"$\varepsilon$-first",
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
        "KL-UCB-SC+",
        "UCB-SC+",
        "UCB-B2",
        r"$\omega$-UCB"
    ]

    # 0: approaches perform similarly
    # 1: approach in row was better than approach in column
    # -1: approach in column was better than approach in row
    # -2: approaches were not compared
    matrix = [
        [0, -1, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -2, -2],
        [1, 0, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -2, -2],
        [-2, 1, 0, 0, -1, -1, -2, 0, 0, 0, 0, -1, -1, -2, -2],
        [-2, -2, -2, 0, 0, -1, -2, -1, -1, -1, -1, 0, -1, -2, -2],
        [-2, -2, 1, -2, 0, 0, -2, 0, 0, 0, 0, 0, 0, -2, -1],
        [1, 1, 1, 1, -2, 0, -1, -1, -1, -1, -1, 0, 0, -2, -1],
        [1, 1, -2, -2, -2, 1, 0, -1, -2, -1, -1, -2, -2, -2, -1],
        [1, -2, -2, 1, -2, 1, 1, 0, 1, -1, -1, 0, 0, -2, -1],
        [1, -2, -2, 1, -2, 1, -2, -1, 0, -1, -1, 0, 0, -2, -1],
        [1, -2, -2, 1, -2, 1, 1, 1, 1, 0, 0, 0, 0, -2, -1],
        [1, -2, -2, 1, -2, 1, 1, 1, 1, 0, 0, 0, 0, -2, -1],
        [-2, 0, 1, 0, -2, 0, -2, -2, -2, -2, -2, 0, -2, -2, -2],
        [-2, -2, 1, 1, -2, 0, -2, -2, -2, -2, -2, -2, 0, -2, -1],
        [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 0, -1],
        [-2, -2, -2, -2, 1, 1, 1, 1, 1, 1, 1, -2, 1, 1, 0]
    ]

    matrix = np.array(matrix)

    ###### This corrects some of the information that got messed up while extending the matrix above between iterations of the paper
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            if matrix[row][col] == -2:
                matrix[col][row] = -2

    matrix_inv = [matrix[-i - 1] for i in range(len(matrix))]
    mask = np.triu(np.ones_like(matrix_inv))
    mask = np.array([mask[-(i + 1)] for i in range(len(mask))])
    approaches_inv = [approaches[-(i + 1)] for i in range(len(approaches))]

    first_row = matrix[7, :].copy()
    matrix[7, :] = matrix[8, :]
    matrix[8, :] = first_row

    col_7 = matrix[:, 7].copy()
    matrix[:, 7] = matrix[:, 8]
    matrix[:, 8] = col_7
    ######

    show_full = True
    remove_ours = True
    grays = sns.color_palette("Greys", n_colors=101)
    grays = [grays[30], grays[10]]
    palette = list(sns.color_palette("vlag_r", n_colors=101))
    palette = [palette[27]] + [grays[-1]] + [palette[87]]
    palette = palette if show_full else palette[1:]
    palette = [grays[0]] + palette

    if remove_ours:
        approaches = approaches[:-1]
        matrix = np.array(matrix)[:-1, :-1]
        if remove_ours and not show_full:
            raise NotImplementedError

    empty = [
        "" for _ in range(len(approaches))
    ]

    df = pd.DataFrame(np.array(matrix), columns=approaches, index=approaches)
    ax = sns.heatmap(df, cmap=palette, linewidth=1, cbar=False, square=True)
    sns.scatterplot(x=.5 + np.argwhere(df.to_numpy().T == 1).T[0],
                    y=.5 + np.argwhere(df.to_numpy().T == 1).T[1],
                    ax=ax, facecolor="none", edgecolor="white", marker="^")
    sns.scatterplot(x=.5 + np.argwhere(df.to_numpy().T == -1).T[0],
                    y=.5 + np.argwhere(df.to_numpy().T == -1).T[1],
                    ax=ax, facecolor="none", edgecolor="white", marker="v")
    sns.scatterplot(x=.5 + np.argwhere(df.to_numpy().T == 0).T[0],
                    y=.5 + np.argwhere(df.to_numpy().T == 0).T[1],
                    ax=ax, edgecolor="gray", facecolor="none", marker="o")
    sns.scatterplot(x=.5 + np.argwhere(df.to_numpy().T == -2).T[0],
                    y=.5 + np.argwhere(df.to_numpy().T == -2).T[1],
                    ax=ax, color="white", marker="_")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.set_label_position('top')
    ax.yaxis.set_label_position('left')
    ax.xaxis.tick_top()
    for item in ax.get_xticklabels():
        item.set_rotation(90)
    for item in ax.get_yticklabels():
        item.set_rotation(0)
    ax.set_yticklabels(empty)
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    plt.gcf().set_size_inches(cm2inch((6, 8)))
    plt.tight_layout(pad=.5)
    plt.subplots_adjust(bottom=-0.07)
    plt.savefig(os.path.join(os.getcwd(), "figures", "dominance_table.pdf"))
    plt.show()
