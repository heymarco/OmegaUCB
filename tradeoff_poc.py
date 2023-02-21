import os

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from scipy.optimize import curve_fit
from sklearn.datasets import fetch_california_housing, load_digits, fetch_openml
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt


import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = rho'\usepackage{times}'
mpl.rcParams['text.latex.preamble'] = rho'\usepackage{nicefrac}'
mpl.rc('font', family='serif')


def noise_level(t, N, M0):
    return (1 - M0) * np.exp(-N * t)


def quality_function(t, N, M0):
    return 1 - noise_level(t, N, M0)


def add_noise(y, n_classes: int, prob: float):
    if np.random.uniform() < prob:
        y_new = np.random.choice(range(n_classes))
        while y_new == y:
            y_new = np.random.choice(range(n_classes))
        y = y_new
    return y


def conv(x, M0, q_t, y_50_at_n1, n):
    M0 = 1 / 10
    a = q_t - M0
    b = y_50_at_n1
    return M0 + a * np.power(x, n) / (b + np.power(x, n))


MNIST = 554
MUSHROOM = 24


if __name__ == '__main__':
    # 0. load data
    split = 0.8
    N = .08
    clean_data, clean_labels = fetch_openml(data_id=MNIST, return_X_y=True, as_frame=False)
    not_na_rows = np.any(np.isnan(clean_data), axis=1) == False
    clean_data = clean_data[not_na_rows]
    clean_labels = clean_labels[not_na_rows]
    shuffled_indices = np.arange(len(clean_labels))
    np.random.shuffle(shuffled_indices)
    clean_data = clean_data[shuffled_indices]
    clean_labels = clean_labels[shuffled_indices]
    n_train = int(len(clean_data) * split)
    x_train, x_test = clean_data[:n_train], clean_data[n_train:]
    y_train, y_test = clean_labels[:n_train], clean_labels[n_train:]

    # 1. create noisy data
    B = len(y_train) * .1 * 10
    t_0 = 10
    n_0 = int(0.05 * B / t_0)  # 5% of budget
    M0 = 1 / len(np.unique(clean_labels))
    relative_noise = noise_level(t_0, N=N, M0=M0)
    print("Budget = {}".format(B))
    print("Noise level = {}".format(relative_noise))
    n_classes = len(np.unique(clean_labels))
    y_test_noisy = np.array([
        add_noise(y, n_classes, relative_noise) for y in y_test
    ])
    y_train_noisy = np.array([
        add_noise(y, n_classes, relative_noise) for y in y_train
    ])

    # 2. fit model
    x_initial = x_train[:n_0]
    y_initial = y_train_noisy[:n_0]
    model = RandomForestClassifier(n_estimators=10)
    performance = []
    for frac in tqdm(range(1, n_0, 5)):
        acc_noisy = []
        acc_clean = []
        frac = frac / n_0
        n_samples = int(n_0 * frac)
        for _ in range(3):
            train_indices = np.random.choice(range(n_0),
                                             n_samples, replace=False)
            test_indices = np.delete(np.arange(n_0), train_indices)
            x_sample = x_initial[train_indices]
            y_sample = y_initial[train_indices]
            x_sample_test = x_initial[test_indices]
            y_sample_test_noisy = y_initial[test_indices]
            model.fit(x_sample, y_sample)
            s = model.score(x_test, y_test_noisy)
            s_clean = model.score(x_test, y_test)
            acc_noisy.append(s)
            acc_clean.append(s_clean)
        performance.append([
            n_samples,
            np.mean(acc_noisy),
            np.mean(acc_clean)
        ])

    # 3. fit curve
    performance = np.array(performance)
    x_axis_vals = performance[:, 0]
    y_axis_vals = performance[:, 1]
    M0, q, b, n = curve_fit(conv, x_axis_vals, y_axis_vals,
                             p0=[M0, .5, 500, 1], bounds=(0, [.5, 1, np.inf, np.inf]))[0]
    print("M0 = {}".format(M0))
    print("q_t = {}".format(q))
    q = min(q, .999)
    print("n_50 = {}".format(b))

    # 4. compute Q based on fitted curve
    # (_, N_emp), _ = curve_fit(quality_function, [0, t_0], [M0, q])
    N_emp = -np.log((1 - q) / (1 - M0)) / t_0
    print("N_emp = {}".format(N_emp))

    # n_data_with_budget = int(B / t_0)
    model.fit(x_train, y_train_noisy)
    max_performance_noisy = model.score(x_test, y_test_noisy)
    max_performance_clean = model.score(x_test, y_test)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.lineplot(x=performance[:, 0], y=conv(performance[:, 0], M0, q, b, n), ax=ax)
    sns.scatterplot(x=performance[:, 0], y=performance[:, 1], ax=ax, label="Eval. on noisy")
    sns.scatterplot(x=performance[:, 0], y=performance[:, 2], ax=ax, color="black", label="Eval. on clean")
    plt.axhline(q, lw=.5, color="orange", label="Label quality")
    plt.axhline(max_performance_noisy, color="blue", lw=.5, label="Max. performance noisy")
    plt.axhline(max_performance_clean, color="black", lw=.5, label="Max. performance clean")
    ax.set_ylim(bottom=0)
    plt.legend()
    plt.show()

    def explore(t_prev, t_now, left: bool, left_subtree: bool):
        if left or (not left and left_subtree):
            factor = .5
        else:
            factor = 2
        diff = abs(t_now - t_prev)
        if left:
            return t_now - diff * factor
        else:
            return t_now * factor

    t_opt = t_0
    s_opt = q
    t_prev = 0
    acc_sim = []
    label_quality = []
    percent_convergence = []

    t_max = 151
    r_sim = range(1, t_max)
    for i in r_sim:
        t = i
        q = 1 - noise_level(t, N_emp, M0)
        num_data = int(B / t)
        label_quality.append(q)
        s = conv(num_data, M0, q, b, n)
        percent_convergence.append(s / q)
        acc_sim.append(s)

    t_opt = np.argmax(acc_sim)

    x_range = np.arange(100)
    plt.plot(x_range, noise_level(x_range, N, M0))
    plt.plot(x_range, noise_level(x_range, N_emp, M0))
    plt.show()

    print("s_opt = {}".format(np.max(acc_sim)))
    print("t_opt = {}".format(t_opt))

    acc_noisy = []
    acc_clean = []
    r_eval = range(1, t_max, 1)
    for t in tqdm(r_eval):
        num_data = int(B / t)
        relative_noise = noise_level(t, N=N, M0=M0)
        noisy = []
        clean = []
        for _ in range(10):
            num_data = int(B / t)
            shuffled_indices = np.arange(num_data)
            np.random.shuffle(shuffled_indices)
            x_sample = x_train[shuffled_indices]
            y_sample = np.array([
                add_noise(y, n_classes, relative_noise) for y in y_train[shuffled_indices]
            ])
            y_test_noisy = np.array([add_noise(y, n_classes, relative_noise) for y in y_test])
            model.fit(x_sample, y_sample)
            noisy.append(model.score(x_test, y_test_noisy))
            clean.append(model.score(x_test, y_test))
        acc_noisy.append(np.mean(noisy))
        acc_clean.append(np.mean(clean))

    print("t_opt real = {}".format(r_eval[np.argmax(acc_clean)]))

    # plt.plot(r_sim, label_quality, label=rho"$q_{max}$", color="black", linestyle='dashed')
    # plt.plot(r_sim, percent_convergence, label="model convergence", color="black", linestyle='dotted')
    plt.plot(r_eval, acc_clean, label="acc. on clean data")
    plt.plot(r_eval, acc_noisy, label="acc. on noisy data")
    plt.plot(r_sim, acc_sim, label="predicted acc.")
    plt.legend()
    plt.xlabel(rho"$t$")
    plt.ylabel(rho"Accuracy")
    plt.gcf().set_size_inches((7, 4))
    plt.tight_layout(pad=.5)
    plt.savefig(os.path.join(os.getcwd(), "figures", "example_result.pdf"))
    plt.show()


