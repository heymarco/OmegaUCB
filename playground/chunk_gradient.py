import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from tqdm import tqdm

from components.path import PathElement, Path
from components.data import SymmetricCrowdsourcingOracle
from util import MUSHROOM, MNIST


def labeling(data, path, t, n, oracle):
    x = []
    y = []
    for _ in range(n):
        y.append(oracle.query(t))
        y_index = oracle.queried_index()
        x.append(data[y_index])
    x = np.vstack(x)
    y = np.vstack(y)
    element = PathElement(data=(x, y), time=t)
    path.elements.append(element)


if __name__ == '__main__':
    split = 0.8
    n = 100
    p = 0.0
    clean_data, clean_labels = fetch_openml(data_id=MNIST, return_X_y=True, as_frame=False)
    not_na_rows = np.any(np.isnan(clean_data), axis=1) == False
    clean_data = clean_data[not_na_rows]
    clean_labels = clean_labels[not_na_rows]
    shuffled_indices = np.arange(len(clean_labels))
    np.random.shuffle(shuffled_indices)
    clean_data = clean_data[shuffled_indices]
    clean_labels = clean_labels[shuffled_indices]
    clean_labels = LabelEncoder().fit_transform(clean_labels)
    n_train = int(len(clean_data) * split)
    x_train, x_test = clean_data[:n_train], clean_data[n_train:]
    y_train, y_test = clean_labels[:n_train], clean_labels[n_train:]
    path = Path()
    ps = [0.0, 0.1, 0.2, 0.6]
    oracle = SymmetricCrowdsourcingOracle(y=y_train, p=p, seed=0)

    results = []
    for r in tqdm(range(50)):
        labeling(x_train, path, t=1, n=n, oracle=oracle)
        if len(path) < 2:
            continue
        model = DecisionTreeClassifier()
        x, y = path.data()
        model.fit(x, y)
        score_now = model.score(x_test, y_test)
        for i in range(1):
            if i < len(path):
                subpath = path.subpath_without_index(i)
                x, y = subpath.data()
                model.fit(x, y)
                score = model.score(x_test, y_test)
                results.append([r, i, score, score_now])
    df = pd.DataFrame(results, columns=["Round", "Element", "Score", "Score now"])
    df["Diff"] = df["Score now"] - df["Score"]
    # sns.lineplot(data=df, x="Round", y="Score")
    # sns.lineplot(data=df, x="Round", y="Score now")
    plt.axhline(df["Diff"].mean())
    sns.lineplot(data=df, x="Round", y="Diff")
    plt.tight_layout(pad=.5)
    plt.show()

