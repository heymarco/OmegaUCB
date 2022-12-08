from typing import List

from components.experiments.experiments import UniformArmsExperiment, FacebookAdDataExperiment
from components.experiments.abstract import Experiment

from util import save_df


def execute_experiment(exp: Experiment, arms: List[int], num_reps: int):
    df = exp.run(arms, num_reps=num_reps)
    save_df(df, exp.name)


if __name__ == '__main__':
    arms = [100, 10, 50]
    n_reps = 100
    n_steps = int(1.5e5)

    c_min_experiment = UniformArmsExperiment("uniform_vary_costs", num_steps=n_steps)
    execute_experiment(c_min_experiment, arms, num_reps=n_reps)

    facebook_experiment = FacebookAdDataExperiment("facebook_ads", num_steps=n_steps)
    execute_experiment(facebook_experiment, arms=[0], num_reps=n_reps)

