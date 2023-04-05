# Budgeted Multi-Armed Bandits with Asymmetric Confidence Intervals

This repository contains code for the paper "Budgeted Multi-Armed Bandits with Asymmetric Confidence Intervals".

## Abstract

We focus on the stochastic Budgeted Multi-Armed Bandit (MAB) problem, where a player chooses from $K$ arms with unknown expected rewards and costs. The goal is to maximize the cumulative reward within a given budget by choosing the arm with the highest reward-cost ratio as frequently as possible. We propose a new upper confidence bound (UCB) sampling policy, $\omega$-UCB, that uses asymmetric confidence intervals. The intervals
scale with the distance between the sample mean and the bounds of a random variable. This leads to a more accurate estimation of the reward-cost ratio and resolves multiple issues of existing UCB sampling policies. We prove that $\omega$-UCB has logarithmic regret and our experiments show that it outperforms the current state of the art, both in synthetic and real settings.

## Reproducing our results

### Installing the dependencies
We provide an `environment.yml` file for recreating our conda environment. The easiest way to recreate it is to execute `conda env create -f environment.yml` from the OmegaUCB-directory. In case you run into errors, we advise you to inspect the `environment.yml` file and install missing packages manually. 

### Running the experiments
To reproduce our experimental results, execute `experiment_runner.py`. The script will use all cores on your machine. On our 30-core machine, the experiments take approximately 24 hours. 

### Graphs
- To reproduce Figure 1, run `tightness_coverage_comparison.py`.
- To reproduce Figure 2, run `dominance_matrix.py`.
- For the graphs in Figure 3, run the scripts `mining/bernoulli.py`, `mining/beta.py`, `mining/multinomial.py`, `mining/advertisement_data.py`, and `ad_data_arms_histogram.py`.
- For the graphs in Figure 4, run `mining/sensitivity_study.py`

## Applying $\omega$-UCB in other projects

To support further research, we will link another repository that contains  $\omega$-UCB as a pip-package after paper acceptance.
