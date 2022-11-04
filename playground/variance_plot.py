import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import beta

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{times}'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{nicefrac}'
mpl.rc('font', family='serif')


if __name__ == '__main__':
    def variance(ax, bx, ay):
        n = ax + bx
        frac1 = ax ** 2 / n ** 2
        frac2 = (n - 1) / (ay - 1)
        frac3 = (n - 2) / (ay - 2)
        frac4 = (n - 1) / (ay - 1)
        result = frac1 * frac2 * (frac3 - frac4)
        return result

    def mean(ax, bx, ay):
        n = ax + bx
        return ax / n * (n - 1) / (ay - 1)

    def variance2(ax, bx, ay):
        n = ax + bx
        frac1 = beta(ax + 2, bx) / beta(ax, bx)
        frac2 = beta(ay - 2, n - ay) / beta(ay, n - ay)
        second_moment = frac1 * frac2
        first_moment = beta(ax + 1, bx) * beta(ay - 1, n - ay) / (beta(ax, bx) * beta(ay, n - ay))
        return second_moment - first_moment ** 2

    # n = 100
    # bx = np.linspace(3, n)
    # by = np.linspace(3, n)
    # a_x = n - bx
    # a_y = n - by
    # X, Y = np.meshgrid(bx, by)
    #
    # ax = plt.axes(projection='3d')
    # ax.contour3D(X, Y, variance2(a_x, X, Y), 50, cmap='binary')
    # plt.show()

    fig, axes = plt.subplots(1, 3)
    steps = 20
    # Setting 0
    n = 20
    ax = n / 2
    bx = n - ax
    ay = np.arange(steps, 3 * steps + 1) / steps
    axes[0].plot(ay, np.sqrt(variance2(ax, bx, ay)))
    # axes[0].plot(ay, variance(ax, bx, ay))
    axes[0].plot(ay, mean(ax, bx, ay))
    axes[0].set_xlabel(r"$1 \leq \beta^c < 3$ for $n={}$".format(n))

    # Setting 1
    n = 20
    ax = n / 2
    bx = n - ax
    ay = np.arange(3 * steps, n * steps + 1) / steps
    axes[1].plot(ay, np.sqrt(variance2(ax, bx, ay)))
    # axes[1].plot(ay, variance(ax, bx, ay))
    axes[1].plot(ay, mean(ax, bx, ay))
    axes[1].set_xlabel(r"$3\leq \beta^c$ for $n={}$".format(n))

    # Setting 2
    n = 100
    ax = n / 2
    bx = n - ax
    ay = np.arange(3 * steps, n * steps / 4 + 1) / steps
    axes[2].plot(ay, np.sqrt(variance2(ax, bx, ay)), label="Std.")
    # axes[2].plot(ay, variance(ax, bx, ay))
    axes[2].plot(ay, mean(ax, bx, ay), label="Mean")
    axes[2].set_xlabel(r"$\beta^c$ for $n={}$".format(n))

    plt.suptitle(r"Standard deviation and mean for reward-cost-ratio for $\alpha^r = \beta^r$")
    plt.legend()
    plt.gcf().set_size_inches(6, 1.7)
    plt.tight_layout(pad=.5)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", "beta_std.pdf"))
    plt.show()

    fig, axes = plt.subplots(1, 3)
    steps = 20
    # Setting 0
    n = 20
    ax = n / 2
    bx = n - ax
    ay = np.arange(steps, 3 * steps + 1) / steps
    axes[0].plot(ay, variance2(ax, bx, ay), color="green")
    # axes[0].plot(ay, variance(ax, bx, ay))
    axes[0].plot(ay, mean(ax, bx, ay))
    axes[0].set_xlabel(r"$1\leq\beta^c < 3$ for $n={}$".format(n))

    # Setting 1
    n = 20
    ax = n / 2
    bx = n - ax
    ay = np.arange(3 * steps, n * steps + 1) / steps
    axes[1].plot(ay, variance2(ax, bx, ay), color="green")
    # axes[1].plot(ay, variance(ax, bx, ay))
    axes[1].plot(ay, mean(ax, bx, ay))
    axes[1].set_xlabel(r"$3\leq \beta^c$ for $n={}$".format(n))

    # Setting 2
    n = 100
    ax = n / 2
    bx = n - ax
    ay = np.arange(3 * steps, n * steps / 4 + 1) / steps
    axes[2].plot(ay, variance2(ax, bx, ay), label="Variance", color="green")
    # axes[2].plot(ay, variance(ax, bx, ay))
    axes[2].plot(ay, mean(ax, bx, ay), label="Mean")
    axes[2].set_xlabel(r"$\beta^c$ for $n={}$".format(n))

    plt.suptitle(r"Variance and mean for reward-cost-ratio for $\alpha^r = \beta^r$")
    plt.legend()
    plt.gcf().set_size_inches(6, 1.7)
    plt.tight_layout(pad=.5)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", "beta_variance.pdf"))
    plt.show()