import matplotlib.pyplot as plt
from scipy import stats
from scipy import special
from scipy import integrate
import numpy as np


def exp_value_beta(a, b):
    return a / (a + b)


def reg_x(x, n, exp=2):
    return 0.5 + (1 - 1 / n ** exp) * (x - 0.5)


def exp_value_beta_reg(a, b):
    return integrate.quad(
        lambda x: reg_x(x, a + b) * stats.beta.pdf(x, a, b), 0, 1
    )[0]


def exp_value_inverse_beta(a, b):
    return integrate.quad(
        lambda x: 1 / x * stats.beta.pdf(x, a, b), 0, 1
    )[0]


def exp_value_inverse_beta_reg(a, b):
    return integrate.quad(
        lambda x: 1 / reg_x(x, a + b) * stats.beta.pdf(x, a, b), 0, 1
    )[0]


def exp_value_beta_squared(a, b):
    return integrate.quad(
        lambda x: x ** 2 * stats.beta.pdf(x, a, b), 0, 1
    )[0]


def exp_value_beta_squared_reg(a, b):
    return integrate.quad(
        lambda x: reg_x(x, a + b) ** 2 * stats.beta.pdf(x, a, b), 0, 1
    )[0]


def exp_value_inverse_beta_squared(a, b):
    return integrate.quad(
        lambda x: x ** -2 * stats.beta.pdf(x, a, b), 0, 1
    )[0]


def exp_value_inverse_beta_squared_reg(a, b):
    return integrate.quad(
        lambda x: reg_x(x, a + b) ** -2 * stats.beta.pdf(x, a, b), 0, 1
    )[0]


def ratio_variance(a1, b1, a2, b2):
    res1 = exp_value_beta_squared(a1, b1)
    res2 = exp_value_inverse_beta_squared(a2, b2)
    res3 = exp_value_beta(a1, b1) ** 2
    res4 = exp_value_inverse_beta(a2, b2) ** 2
    return res1 * res2 - res3 * res4


def reg_ratio_variance(a1, b1, a2, b2):
    res1 = exp_value_beta_squared_reg(a1, b1)
    res2 = exp_value_inverse_beta_squared_reg(a2, b2)
    res3 = exp_value_beta_reg(a1, b1) ** 2
    res4 = exp_value_inverse_beta_reg(a2, b2) ** 2
    return res1 * res2 - res3 * res4


def ratio_kth_raw_moment(a1, b1, a2, b2, k):
    top = special.beta(a1 + k, b1) * special.beta(a2 - k, b2)
    bottom = special.beta(a1, b1) * special.beta(a2, b2)
    return top / bottom


def variance(a1, b1, a2, b2):
    second_raw_moment = ratio_kth_raw_moment(a1, b1, a2, b2, 2)
    squared_first_moment = ratio_kth_raw_moment(a1, b1, a2, b2, 1) ** 2
    return second_raw_moment - squared_first_moment


if __name__ == '__main__':
    r = np.arange(200, 1000, 5) / 100
    original = [ratio_variance(val, val, val, val) for val in r]
    regularized = [reg_ratio_variance(val, val, val, val) for val in r]

    mean = [ratio_kth_raw_moment(val, val, val, val, k=1) for val in r]

    plt.plot(2 * r, mean, label="mean")
    plt.plot(2 * r, np.sqrt(original), label="std")
    plt.plot(2 * r, np.sqrt(regularized), label="reg. std")
    # plt.plot(2 * r, np.array(mean) ** 2 / np.array(original), label="snr original")
    # plt.plot(2 * r, np.array(mean) ** 2 / np.array(regularized), label="snr regularized")
    plt.legend()
    plt.yscale("log")
    plt.show()
