import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.special import digamma


if __name__ == '__main__':
    a_factor = 0.1
    b_factor = 0.1
    a_values = (np.arange(2000) * a_factor).astype(int)
    b_values = (np.arange(2000) * b_factor).astype(int)
    p = a_factor / (a_factor + b_factor)

    gt_entropy = - (p * np.log(p) + (1 - p) * np.log(1 - p))

    def x_ln_x(a, b):
        factor1 = a / (a + b)
        factor2 = digamma(a + 1) - digamma(a + b + 1)
        return factor1 * factor2

    def expected_entropy(alpha: int, beta: int):
        add1 = x_ln_x(alpha, beta)
        add2 = x_ln_x(beta, alpha)
        return -(add1 + add2)

    def beta_entropy(alpha: int, beta: int):
        return stats.beta.entropy(alpha, beta)

    e_entropy = []
    for a, b in zip(a_values, b_values):
        e_entropy.append(expected_entropy(a + 1, b + 1))

    plt.axhline(gt_entropy, label="True entropy")
    plt.plot(a_values + b_values, e_entropy, label="Expected entropy", color="orange"
             # , marker="o"
             )
    # plt.plot(a_values + b_values, beta_entropy(a_values, b_values), color="green", label="beta entropy")
    plt.legend()
    plt.show()
