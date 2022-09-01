import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats

if __name__ == '__main__':
    def hoeffding_ci(n: int, alpha=0.05):
        return np.sqrt(-1 / (2 * n) * np.log(alpha / 2))

    def markov_ci(n: int, alpha=0.05):
        return 1 / (alpha * n)

    def wilson_ci(n: int, ns: int, alpha=0.05):
        nf = n - ns
        z = stats.norm.interval(1 - alpha)[1]
        return 2 / (n + z ** 2) * np.sqrt((ns * nf) / n + z ** 2 / 4)

    def combined_ci(n: int, alpha=0.01):
        return min(markov_ci(n, alpha), hoeffding_ci(n, alpha))

    def cumulative_ci(n: int):
        return np.sum([hoeffding_ci(i) for i in range(1, n)])

    def beta_mean(alpha, beta):
        return alpha / (alpha + beta)

    trial = [(hoeffding_ci(i), wilson_ci(i, int(i / 2))) for i in range(2, 100)]
    print(trial)

    ns = [3, 10, 30, 100, 1000]
    ps = np.arange(5, 50) / 50
    result = []
    n_samples = 100
    mean_reward = 0.5
    metric_types = ["mean", "mean", "mean", "mean", "std.",
                    "std.", "std.", "E[r/c] / bias", "E[r/c] / bias", "E[r/c] / bias",
                    "mean + bias", "mean + bias", "mean + bias",
                    "E[r/c] / noise", "E[r/c] / noise", "E[r/c] / noise"]
    methods = ["ground truth", "original", "ci", "1 / sqrt(n)",
               "original", "ci", "1 / sqrt(n)",
               "original", "ci", "1 / sqrt(n)",
               "original", "ci", "1 / sqrt(n)",
               "original", "ci", "1 / sqrt(n)"]
    for p in tqdm(ps):
        for n in ns:
            for s in range(n_samples):
                bernoulli_trials = np.random.uniform(size=n) < p
                rewards = np.random.uniform(size=n) < mean_reward

                original_alpha = np.sum(bernoulli_trials)
                original_beta = n - np.sum(bernoulli_trials)
                original_alpha_rew = np.sum(rewards)
                original_beta_rew = n - np.sum(rewards)

                sqrt_alpha = min(n, np.sum(bernoulli_trials) + n * wilson_ci(n, np.sum(bernoulli_trials)))
                sqrt_beta = n - min(n, np.sum(bernoulli_trials) + n * wilson_ci(n, np.sum(bernoulli_trials)))
                sqrt_alpha_rew = min(n, np.sum(rewards) + n * wilson_ci(n, np.sum(bernoulli_trials)))
                sqrt_beta_rew = n - min(n, np.sum(rewards) + n * wilson_ci(n, np.sum(bernoulli_trials)))

                ci_alpha = min(n, np.sum(bernoulli_trials) + n * hoeffding_ci(n))
                ci_beta = n - sqrt_alpha
                ci_alpha_rew = min(n, np.sum(rewards) + n * hoeffding_ci(n))
                ci_beta_rew = n - sqrt_alpha_rew

                samples_original = np.random.beta(original_alpha + 1, original_beta + 1, size=1000)
                samples_ci = np.random.beta(ci_alpha + 1, ci_beta + 1, size=1000)
                samples_sqrt = np.random.beta(sqrt_alpha + 1, sqrt_beta + 1, size=1000)

                top = mean_reward
                samples_top_original = np.random.beta(original_alpha_rew + 1, original_beta_rew + 1, size=1000)
                samples_top_ci = np.random.beta(ci_alpha_rew + 1, ci_beta_rew + 1, size=1000)
                samples_top_sqrt = np.random.beta(sqrt_alpha_rew + 1, sqrt_beta_rew + 1, size=1000)

                ratio_original = samples_top_original / samples_original
                ratio_ci = samples_top_ci / samples_ci
                ratio_sqrt = samples_top_sqrt / samples_sqrt

                std_original = np.std(ratio_original, axis=-1)
                std_ci = np.std(ratio_ci, axis=-1)
                std_sqrt = np.std(ratio_sqrt, axis=-1)

                mean_original = np.mean(ratio_original, axis=-1)
                mean_ci = np.mean(ratio_ci, axis=-1)
                mean_sqrt = np.mean(ratio_sqrt, axis=-1)

                bias_original = (mean_original - top / p)
                bias_ci = (mean_ci - top / p)
                bias_sqrt = (mean_sqrt - top / p)

                snr_original = mean_original / std_original
                snr_ci = mean_ci / std_ci
                snr_sqrt = mean_sqrt / std_sqrt

                original_mean_plus_bias = mean_original + bias_original
                ci_mean_plus_bias = mean_ci + bias_ci
                sqrt_mean_plus_bias = mean_sqrt + bias_sqrt

                numbers = [top / p, mean_original, mean_ci, mean_sqrt,
                           std_original, std_ci, std_sqrt,
                           bias_original, bias_ci, bias_sqrt,
                           original_mean_plus_bias, ci_mean_plus_bias, sqrt_mean_plus_bias,
                           snr_original, snr_ci, snr_sqrt]
                for num, t, method in zip(numbers, metric_types, methods):
                    result.append([n, p, num, t, method])

    df = pd.DataFrame(result, columns=["n", "average cost", "value", "type", "method"])
    df = df[df["type"] != "mean + bias"]
    df = df.groupby(["n", "average cost", "type", "method"]).mean()

    sns.relplot(data=df, x="average cost", y="value",
                hue="method", col="n", row="type",
                kind="line",
                height=2, aspect=1.5, ci=False,
                facet_kws={"sharey": False})
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", "b-mab-snr.pdf"))
    plt.show()

    #
    #
    # top = np.random.beta(5, 5, size=(1000, 100))
    # bottom = [np.random.beta(a + 1, b + 1, size=(1000, 100)) for (a, b) in zip(alpha_range, beta_range)]
    #
    # ratio = [top / b for b in bottom]
    # signal = [np.mean(r, axis=0) for r in ratio]
    # variance = [np.std(r, axis=0) for r in ratio]
    # snr = [v / s for (v, s) in zip(variance, signal)]
    # snr = [np.mean(s) for s in snr]
    #
    # x = [beta_mean(a + 1, b + 1) for (a, b) in zip(alpha_range, beta_range)]
    #
    # plt.plot(x, snr)
    # plt.legend()
    # plt.xlabel("E[denominator]")
    # plt.ylabel("Signal to noise ratio")
    # plt.tight_layout(pad=.5)
    # plt.savefig(os.path.join(os.getcwd(), "..", "figures", "b-mab-snr.pdf"))
    # plt.show()