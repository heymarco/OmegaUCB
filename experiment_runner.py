import os

from components.experiments.experiments import UniformArmsExperiment, prepare_df2
import matplotlib.pyplot as plt
import seaborn as sns

from components.bandit_logging import *


def save_df(df: pd.DataFrame, name: str):
    path = os.path.join(os.getcwd(), "results", name + ".csv")
    df.to_csv(path, index=False)


def load_df(name: str):
    path = os.path.join(os.getcwd(), "results", name + ".csv")
    return pd.read_csv(path)


if __name__ == '__main__':
    arms = [100, 10]
    experiment = UniformArmsExperiment("uniform_vary_costs", num_steps=1e5)
    df = experiment.run(arms, num_reps=30)
    save_df(df, experiment.name)
    # df = load_df(experiment.name)
    # df = prepare_df2(df)
    # g = sns.relplot(data=df, x=NORMALIZED_BUDGET, y=REGRET, hue=APPROACH, col=OPTIMAL_COST,
    #             col_wrap=3, kind="line", markers="o",
    #             facet_kws={"sharey": False}, err_style=None)
    # g.set(yscale="log")
    # # g.set(xscale="log")
    # plt.show()
