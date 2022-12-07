from typing import List

from components.experiments.experiments import UniformArmsExperiment, FacebookAdDataExperiment
from components.experiments.abstract import Experiment

from util import save_df


def execute_experiment(exp: Experiment, arms: List[int], num_reps: int):
    df = exp.run(arms, num_reps=num_reps)
    save_df(df, exp.name)


if __name__ == '__main__':
    arms = [10, 100]
    n_reps = 100

    facebook_experiment = FacebookAdDataExperiment("facebook_ads", num_steps=1e5)
    execute_experiment(facebook_experiment, arms=[0], num_reps=n_reps)

    c_min_experiment = UniformArmsExperiment("uniform_vary_costs", num_steps=1e5)
    execute_experiment(c_min_experiment, arms, num_reps=n_reps)


