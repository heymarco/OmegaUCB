import numpy as np
from scipy.stats import norm


def ci_wilson(x, delta, n, eta=1.0):
    z = norm.interval(1 - delta)[1]
    m, M = 0, 1
    K = eta * (z ** 2)
    A = (n + K)
    B = (2 * n * x + K * (M + m))
    C = (n * x ** 2 + K * M * m)
    low = B / (2 * A) - np.sqrt((B / (2 * A)) ** 2 - C / A)
    high = B / (2 * A) + np.sqrt((B / (2 * A)) ** 2 - C / A)
    return low, x, high


def ci_hoeffding(x, t, n, alpha=1.0):
    eps = alpha * np.sqrt(
        np.log(t-1) / n
    )
    return x - eps, x, x + eps


if __name__ == '__main__':
    t = 10000
    n = 1000
    delta = 1 - np.sqrt(1 - t**-1)
    lcb2, _, ucb2 = ci_wilson(0.1, delta, n)
    lcb1, _, _ = ci_wilson(0.2, delta, n)
    _, _, ucb1 = ci_wilson(0.8, delta, n)
    print(ucb1 / lcb1, ucb2 / lcb2)

    lcb2_mucb, _, ucb2_mucb = ci_hoeffding(0.1, t, n)
    lcb1_mucb, _, _ = ci_hoeffding(0.2, t, n)
    _, _, ucb1_mucb = ci_hoeffding(0.8, t, n)
    print(ucb1_mucb / lcb1_mucb, ucb2_mucb / lcb2_mucb)