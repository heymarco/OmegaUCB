# Budgeted Multi-Armed Bandits with Asymmetric Confidence Intervals

This repository contains code for the paper "Budgeted Multi-Armed Bandits with Asymmetric Confidence Intervals".

## Abstract

We study the stochastic Budgeted Multi-Armed Bandit (MAB) problem, where a player chooses from $K$ arms with unknown expected rewards and costs. The goal is to maximize the total reward under a budget constraint. A player thus seeks to choose the arm with the highest reward-cost ratio as often as possible. Current state-of-the-art policies for this problem have several issues, which we illustrate. To overcome them, we propose a new upper confidence bound (UCB) sampling policy, $\omega$-UCB, that uses asymmetric confidence intervals. These intervals scale with the distance between the sample mean and the bounds of a random variable, yielding a more accurate and tight estimation of the reward-cost ratio compared to our competitors. We show that our approach has logarithmic regret and consistently outperforms existing policies in synthetic and real settings.

## Reproducing our results

### Installing the dependencies
We provide an `environment.yml` file for recreating our conda environment. The easiest way to recreate it is to execute `conda env create -f environment.yml` from the OmegaUCB-directory. In case you run into errors, we advise you to inspect the `environment.yml` file and install missing packages manually. 

### Downloading the advertisement data
Please download the advertisement data from [Kaggle](https://www.kaggle.com/datasets/madislemsalu/facebook-ad-campaign), unpack the zip file, verify that the name of the csv has the name `data.csv` and place the file in the `data` directory of this project.

### Running the experiments
To reproduce our experimental results, execute `experiment_runner.py`. The script will use all cores on your machine. On our 30-core machine, the experiments take approximately 24 hours. 

### Graphs
- To reproduce Figure 1, run `tightness_coverage_comparison.py`.
- To reproduce Figure 2, run `dominance_matrix.py`.
- For the graphs in Figure 3, run the scripts `mining/bernoulli.py`, `mining/beta.py`, `mining/multinomial.py`, `mining/advertisement_data.py`, and `ad_data_arms_histogram.py`.
- For the graphs in Figure 4, run `mining/sensitivity_study.py`

Note that plotting requires a local LaTeX installation. In case LaTeX is not installed or if you encounter other LaTeX-related problems, please try commenting out the following lines in the scripts above:
    
    import matplotlib as mpl

    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
    mpl.rc('font', family='serif')

## Applying $\omega$-UCB in other projects

We will link another repository containing $\omega$-UCB as a pip-package after paper acceptance.
