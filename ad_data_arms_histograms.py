import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from facebook_ad_data_util import get_facebook_ad_data_settings

import matplotlib as mpl

from util import cm2inch

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')

if __name__ == '__main__':
    # Define the columns of the data frame that we'll use later on
    SETTING = "Setting"
    K = "K"
    REWARDS = r"$\mu_k^r$"
    COSTS = r"$\mu_k^c$"
    VALUE = "Value"
    KIND = "Kind"

    # Create the color map for the kde plot
    min_val, max_val = 0.25, 1.0
    n = 10
    orig_cmap = plt.cm.Greys
    colors = orig_cmap(np.linspace(min_val, max_val, n))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("mycmap", colors)

    # Load the settings and create the dataframe
    settings = get_facebook_ad_data_settings()
    df = []
    for i, setting in enumerate(settings):
        rew, cost = setting
        K = len(rew)
        for r, c in zip(rew, cost):
            df.append([
                i, K, r, c
            ])
    df = pd.DataFrame(df, columns=[SETTING, K, REWARDS, COSTS])

    # For each setting, create a KDE plot and save it
    for (k, sett), gdf in df.groupby([K, SETTING]):
        sns.kdeplot(data=gdf, x=COSTS, y=REWARDS,
                    fill=True, clip=((0, 1), (0, 1)),
                    cmap=cmap)
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        num_settings_with_k = len(np.unique(gdf[SETTING]))
        plt.gcf().set_size_inches(cm2inch(18 / 2.8, 5.8 * 0.58))
        plt.ylabel(REWARDS, rotation=0, labelpad=10)
        plt.tight_layout(pad=.5)
        plt.savefig(os.path.join(os.getcwd(), "figures", "histograms", "K{}_{}.pdf".format(k, sett)))
        plt.clf()

    # Also create a KDE plot for all settings
    sns.kdeplot(data=df, x=COSTS, y=REWARDS,
                fill=True, clip=((0, 1), (0, 1))
                )
    plt.title("Distribution of rewards and costs".format())
    plt.gcf().set_size_inches(cm2inch(18 / 2.8, 5.8 * 0.58))
    plt.tight_layout(pad=.5)
    plt.savefig(os.path.join(os.getcwd(), "figures", "histograms", "all.pdf"))
