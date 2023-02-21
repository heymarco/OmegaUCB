import numpy as np
from scipy.stats import norm


def lcb_wilson_generalized(x, n, z, eta=1.0, m=0.0, M=1.0):
    K = eta * (z ** 2)
    A = (n + K)
    B = (2 * n * x + K * (M + m))
    C = (n * x ** 2 + K * M * m)
    solution = B / (2 * A) - np.sqrt((B / (2 * A)) ** 2 - C / A)
    return solution


def lcb_wilson(x, n, z, eta=1.0):
    ns = x * n
    nf = n - ns
    z = np.sqrt(eta) * z
    z2 = np.power(z, 2)
    avg = (ns + 0.5 * z2) / (n + z2)
    ci = z / (n + z2) * np.sqrt((ns * nf) / n + z2 / 4)
    return avg - ci


if __name__ == '__main__':
    delta = 0.05
    eta = 1.0
    M = 2.0
    z = norm.interval(1 - delta)[1]

    n_range = 10
    x = np.arange(1, n_range + 1) / n_range
    n = 1000

    wilson = lcb_wilson(x, n, z, eta)
    wilson_general = lcb_wilson_generalized(x * M, n, z, eta, M=M)

    top = x * M - wilson_general
    bottom = x - wilson
    print(top / bottom)
