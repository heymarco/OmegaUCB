# Budgeted Multi-Armed Bandits with Asymmetric Confidence Intervals

This repository contains code for the paper "Budgeted Multi-Armed Bandits with Asymmetric Confidence Intervals".

## Abstract

We focus on the stochastic Budgeted Multi-Armed Bandit (MAB) problem, where a player chooses from $K$ arms with unknown expected rewards and costs. The goal is to maximize the total reward under a budget constraint. A player thus seeks to choose the arm with the highest reward-cost ratio as often as possible. We identify and illustrate multiple issues in the current state of the art. To overcome these issues, we propose a first-of-its-kind upper confidence bound (UCB) sampling policy, $\omega$-UCB, that uses asymmetric confidence intervals. Our intervals scale with the distance between the sample mean and the bounds of a random variable, leading to a more accurate estimation of the reward-cost ratio compared to our competitors. Our approach enjoys provably logarithmic regret and outperforms the current state of the art consistently in diverse synthetic and real settings. 

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
