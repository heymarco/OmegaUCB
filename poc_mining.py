import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == '__main__':
    path = os.path.join(os.getcwd(), "results", "poc.csv")
    df = pd.read_csv(path)

    for s, gdf in df.groupby("strategy"):
        print(s, gdf["n-data"].iloc[-1])

    for s, gdf in df.groupby("strategy"):
        fraction_correct = np.sum(gdf["corrected-label"] == gdf["true-label"]) / len(gdf)
        print(s, fraction_correct)

    print()

    for s, gdf in df.groupby("strategy"):
        frac_relabeled = np.sum(gdf["should-relabel"]) / len(gdf)
        print(s, frac_relabeled)

    sns.lineplot(data=df, x="index", y="confusion", hue="strategy")
    plt.show()
    sns.lineplot(data=df, x="index", y="n-data", hue="strategy")
    plt.show()
    sns.lineplot(data=df, x="index", y="accuracy", hue="strategy")
    plt.show()
