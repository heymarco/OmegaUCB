import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from facebook_ad_data_util import get_facebook_ad_data_settings

import matplotlib as mpl

from util import cm2inch

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
mpl.rc('font', family='serif')


if __name__ == '__main__':
    SETTING = "Setting"
    K = "K"
    REWARDS = r"$\mu_k^r$"
    COSTS = r"$\mu_k^c$"
    VALUE = "Value"
    KIND = "Kind"

    min_val, max_val = 0.25, 1.0
    n = 10
    orig_cmap = plt.cm.Greys
    colors = orig_cmap(np.linspace(min_val, max_val, n))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("mycmap", colors)

    settings = get_facebook_ad_data_settings(rng=None)
    df = []
    for i, setting in enumerate(settings):
        rew, cost = setting
        K = len(rew)
        for r, c in zip(rew, cost):
            df.append([
                i, K, r, c
            ])
            # df.append([
            #     i, K, c, r"$\mu_k^c$"
            # ])
    df = pd.DataFrame(df, columns=[SETTING, K, REWARDS, COSTS])
    for (k, sett), gdf in df.groupby([K, SETTING]):
        sns.kdeplot(data=gdf, x=COSTS, y=REWARDS,
                    fill=True, clip=((0, 1), (0, 1)),
                    cmap=cmap
                    )
        # plt.title("K={}".format(k))
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        num_settings_with_k = len(np.unique(gdf[SETTING]))
        plt.gcf().set_size_inches(cm2inch((20 / 2.8, 8 * 0.75)))
        plt.ylabel(REWARDS, rotation=0, labelpad=10)
        plt.tight_layout(pad=.5)
        plt.savefig(os.path.join(os.getcwd(), "figures", "histograms", "K{}_{}.pdf".format(k, sett)))
        plt.clf()

    sns.kdeplot(data=df, x=COSTS, y=REWARDS,
                fill=True, clip=((0, 1), (0, 1))
                )
    plt.title("Distribution of rewards and costs".format())
    plt.gcf().set_size_inches(cm2inch((20 / 2.8, 8 * 0.75)))
    plt.tight_layout(pad=.5)
    plt.savefig(os.path.join(os.getcwd(), "figures", "histograms", "all.pdf"))
