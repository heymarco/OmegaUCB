import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from util import linear_cost_function


def load_df():
    path1 = os.path.join(os.getcwd(), "..", "results", "results_baseline.csv")
    path2 = os.path.join(os.getcwd(), "..", "results", "results_bandit.csv")
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df = pd.concat([df1, df2], ignore_index=True)
    df.ffill(inplace=True)
    df["round"] = 0
    df["Grad"] = 0
    df["Cost"] = linear_cost_function(df["t"])
    df["Budget"] = df["Cost"] * df["n"]
    df["Spent budget"] = 0
    for _, gdf in df.groupby(["approach", "rep", "noise-level"]):
        print(gdf["Budget"].cumsum())
        df["Spent budget"].loc[gdf.index] = gdf["Budget"].cumsum()
    for _, gdf in df.groupby(["rep", "approach", "noise-level", "dataset"]):
        df["round"][gdf.index] = np.arange(len(gdf))
        df["Grad"][gdf.index] = gdf["true-score"].diff()
    df["Accuracy"] = df["true-score"]
    return df


def plot_score_over_round():
    df = load_df()
    sns.lineplot(data=df, x="round", y="Accuracy", style="approach", ci=None, hue="noise-level")
    plt.tight_layout(pad=.5)
    plt.show()


def plot_score_over_budget():
    df = load_df()
    # df = df[df["Spent budget"] <= 1000]
    sns.lineplot(data=df, x="Spent budget", y="Accuracy", hue="approach", ci=None, style="noise-level")
    plt.tight_layout(pad=.5)
    plt.show()


def plot_t_over_budget():
    df = load_df()
    budget_per_arm = df.groupby(["t", "approach", "rep", "noise-level"])["Budget"].cumsum()
    df["Budget per arm"] = budget_per_arm
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False, )
    sns.barplot(data=df, x="t", y="Budget per arm", hue="approach", ax=axes[0])
    sns.histplot(data=df, x="t", common_norm=False, stat="percent", multiple="dodge", ax=axes[1], hue="approach")
    plt.tight_layout(pad=.5)
    plt.show()
    mask = df["approach"] == "ThompsonSampling-L"
    mask = np.logical_or(mask, df["approach"] == "ThompsonSampling-U")
    df = df[mask]
    print(df.groupby(["approach", "noise-level"])["t"].mean())
    sns.scatterplot(data=df, x="Spent budget", y="t", style="approach", ci=None)
    plt.tight_layout(pad=.5)
    plt.show()


def plot_grad_over_nlabels():
    df = load_df()
    sns.relplot(data=df, kind="line", x="Spent budget", y="Grad", hue="approach", col="noise-level", ci=None)
    plt.tight_layout(pad=.5)
    plt.show()


def plot_score_over_number_of_labels():
    df = load_df()
    df["Number of labels"] = 0
    for _, gdf in df.groupby(["approach", "rep", "noise-level"]):
        df["Number of labels"].loc[gdf.index] = gdf["n"].cumsum()
    print(df)
    sns.lineplot(data=df, x="Number of labels", y="Accuracy", hue="approach", style="noise-level", ci=None)
    plt.tight_layout(pad=.5)
    plt.show()


if __name__ == '__main__':
    plot_score_over_budget()
    plot_t_over_budget()

