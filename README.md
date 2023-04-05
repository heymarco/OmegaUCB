# Budgeted Multi-Armed Bandits with Asymmetric Confidence Intervals

This repository contains code for the paper "Budgeted Multi-Armed Bandits with Asymmetric Confidence Intervals".

## Abstract

We focus on the Budgeted Multi-Armed Bandit (MAB) problem, where a player chooses from $K$ arms with unknown expected rewards and associated costs. The goal is to maximize the cumulative reward within the given budget by choosing the arm with the highest reward-cost ratio as frequently as possible. We propose a new upper confidence bound (UCB) sampling policy, $\omega$-UCB, that solves issues of existing policies by using asymmetric confidence intervals. These intervals allow a more accurate estimation of the reward-cost ratio. We prove that our approach has logarithmic regret and our experiments show that it outperforms the current state of the art, both in synthetic and real settings.

## Reproducing our results

### Installing the dependencies
We provide an `environment.yml` file to recreate our conda environment. The easiest way to recreate it is to to execute `conda env create -f environment.yml` from the OmegaUCB-directory. In case you run into errors, we advise you to inspect the `environment.yml` file and install missing packages manually. 

### Running the experiments
Simply execute `experiment_runner.py` to run the experiments. Note that the script will use all cores on your machine. On our 30-core machine, the experiments take approximately 24 hours. 

### Graphs
- To reproduce Figure 1, run `tightness_coverage_comparison.py`.
- To reproduce Figure 2, run `dominance_matrix.py`.
- For the graphs in Figure 3, run the scripts `mining/bernoulli.py`, `mining/beta.py`, `mining/multinomial.py`, and `mining/advertisement_data.py`.
- For the graphs in Figure 4, run `mining/sensitivity_study.py`

## Applying $\omega$-UCB in other projects

To support further research, we will link another repository with installation instructions after paper acceptance.
