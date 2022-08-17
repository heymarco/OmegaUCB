import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def load_df():
    path = os.path.join(os.getcwd(), "..", "results", "results_baseline.csv")
    df = pd.read_csv(path)
    df.ffill(inplace=True)
    return df


def plot_score_over_budget():
    df = load_df()
    df["Budget"] = df["t"] * df["n"]
    df["Spent budget"] = 0
    for _, gdf in df.groupby(["rep", "noise-level"]):
        print(gdf["Budget"].cumsum())
        df["Spent budget"].loc[gdf.index] = gdf["Budget"].cumsum()
    sns.lineplot(data=df, x="Spent budget", y="true-score", ci=None, hue="noise-level")
    plt.tight_layout(pad=.5)
    plt.show()


def plot_t_over_budget():
    df = load_df()
    df["Budget"] = df["t"] * df["n"]
    df["Spent budget"] = 0
    for _, gdf in df.groupby(["rep", "noise-level"]):
        print(gdf["Budget"].cumsum())
        df["Spent budget"].loc[gdf.index] = gdf["Budget"].cumsum()
    print(df)
    sns.lineplot(data=df, x="Spent budget", y="t", marker="o", hue="noise-level", ci=None)
    plt.tight_layout(pad=.5)
    plt.show()


if __name__ == '__main__':
    plot_score_over_budget()
    plot_t_over_budget()

