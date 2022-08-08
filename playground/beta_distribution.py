import numpy as np
from scipy.stats import beta
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt


def curve(x, k, q):
    return q * np.power(k, -x)


if __name__ == '__main__':
    r = np.arange(1, 101)
    a = r * 1
    b = r * 1.5
    n = a + b

    b = beta.sf(0.5, a, b)
    (k, q), _ = curve_fit(curve, n, b)
    b_sim = curve(n, k, q)
    print(k, q)

    plt.plot(n, b)
    plt.plot(n, b_sim)
    plt.xlabel("# labels")
    plt.ylabel("P(y=1)")
    plt.tight_layout(pad=.5)
    plt.show()