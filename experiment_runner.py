from typing import List

from components.experiments.experiments import BernoulliExperiment, FacebookBernoulliExperiment, BetaExperiment, \
    FacebookBetaExperiment, BernoulliNormalExperiment, MultinomialExperiment
from components.experiments.abstract import Experiment

from util import save_df


def execute_experiment(exp: Experiment, arms: List[int], num_reps: int):
    df = exp.run(arms, num_reps=num_reps)
    save_df(df, exp.name)


if __name__ == '__main__':
    arms = [
        100,
        10,
        50
    ]
    n_reps = 100
    n_steps = int(1.5e5)

    # facebook_beta = FacebookBetaExperiment("facebook_beta_impressions", num_steps=n_steps)
    # execute_experiment(facebook_beta, arms=[0], num_reps=n_reps)
    #
    # facebook_bernoulli = FacebookBernoulliExperiment("facebook_bernoulli_impressions", num_steps=n_steps)
    # execute_experiment(facebook_bernoulli, arms=[0], num_reps=n_reps)

    # synth_beta = BetaExperiment("synth_beta", num_steps=n_steps)
    # execute_experiment(synth_beta, arms, num_reps=n_reps)
    #
    # synth_bernoulli = BernoulliExperiment("synth_bernoulli_rest", num_steps=n_steps)
    # execute_experiment(synth_bernoulli, arms, num_reps=n_reps)
    #
    synth_bernoulli = MultinomialExperiment("synth_multinomial_rest", num_steps=n_steps)
    execute_experiment(synth_bernoulli, arms, num_reps=n_reps)
    #
    # synth_bernoulli_normal_05 = BernoulliNormalExperiment("synth_bernoulli_normal_05", num_steps=n_steps,
    #                                                       loc=.5, scale=0.05)
    # execute_experiment(synth_bernoulli_normal_05, arms, num_reps=n_reps)
    #
    # synth_bernoulli_normal_75 = BernoulliNormalExperiment("synth_bernoulli_normal_75", num_steps=n_steps,
    #                                                       loc=.75, scale=0.075)
    # execute_experiment(synth_bernoulli_normal_75, arms, num_reps=n_reps)
