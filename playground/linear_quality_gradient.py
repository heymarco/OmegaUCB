import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    def linear(x, a, b):
        return a + b * x


    def sqrt(x):
        return 1 - 1 / np.sqrt(x)


    def michaelis_menten(x, c, d):
        return (x * c) / (x + d)


    def M(x, q_0, q_grad, c, n_50, B):
        t = B / x
        quality = linear(t, 0, q_grad)
        quality = sqrt(t)
        return q_0 + quality * michaelis_menten(x, c, n_50)


    n = np.arange(1, 1000)
    B = 1000
    q_0 = [0.5, 0.6, 0.7, 0.8]
    q_grad = 0.01
    n_50 = 100

    fig, axes = plt.subplots(nrows=len(q_0), sharex=True, sharey=True)

    for ax, q in zip(axes, q_0):
        c = 1 - q
        approx = M(n, q, q_grad, c, n_50, B)
        ax.plot(n, approx)
    plt.tight_layout(pad=.5)
    plt.show()



