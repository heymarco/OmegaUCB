import numpy as np
import matplotlib.pyplot as plt


def chernoff_bound(p, delta, n):
    upper_bound = (1 + delta)*p
    lower_bound = (1 - delta) * p
    return upper_bound, lower_bound


def expected_value(p, n):
    return p


def plot_chernoff_difference(p_values, delta, n, confidence_level=0.95):
    plt.figure(figsize=(10, 6))

    differences = []

    for p in p_values:
        upper_bound, _ = chernoff_bound(p, delta, n)
        expected_val = expected_value(p, n)
        differences.append(upper_bound - expected_val)

    plt.plot(p_values, differences, 'ro-', label='Upper Bound - Expected Value')

    plt.title('Difference between Upper Confidence Bound and Expected Value for Bernoulli Random Variables')
    plt.xlabel('p')
    plt.ylabel('Upper Bound - Expected Value')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage
p_values = np.linspace(0.01, 0.99, 100)  # Generate a range of p values
delta = 0.2
n = 100

plot_chernoff_difference(p_values, delta, n)
