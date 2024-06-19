import os.path

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from util import cm2inch

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')


if __name__ == '__main__':

    def asymmetry(mu, n, z):
        nom = (2 * mu - 1) ** 2 * z ** 2
        denom = 4 * n * mu * (1 - mu) + z ** 2
        return nom / denom

    MU = r"$\mu$"
    ASYMMETRY = "Asymmetry"
    N = r"$n$"
    Z = r"$z$"

    cols = [MU, ASYMMETRY, N, Z]

    z_range = [3]
    n_range = [1, 10, 100, 1000, 10000]
    mu = np.linspace(0, 1, 1000)

    df = []

    for z in z_range:
        for n in n_range:
            asym = asymmetry(mu, n, z)
            for m, a in zip(mu, asym):
                df.append([m, a, n, z])

    df = pd.DataFrame(df, columns=cols)
    g = sns.relplot(df,
                    x=MU, y=ASYMMETRY,
                    hue=N,
                    # col=Z,
                    kind="line",
                    palette=sns.cubehelix_palette(n_colors=len(n_range)))
    sns.move_legend(g, "upper right", ncol=1, frameon=True)

    plt.gcf().set_size_inches(cm2inch(10, 3.5))
    plt.tight_layout(pad=.5)
    plt.subplots_adjust(right=.73, left=.1)
    plt.savefig(os.path.join(os.getcwd(), "figures", "asymmetry.pdf"))
    plt.show()

