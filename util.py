import os.path
import pathlib
from multiprocessing import Pool
from time import sleep

from tqdm import tqdm

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from colors import color_list, get_palette_for_approaches, get_markers_for_approaches, get_linestyles_for_approaches
from components.bandit_logging import *
from approach_names import *

approach_order = {
    BTS: 0,
    UCB_SC_PLUS: 1,
    BUDGET_UCB: 2,
    B_GREEDY: 3,
    CUCB: 4,
    IUCB: 5,
    MUCB: 6,
    OMEGA_UCB_: 7,
    OMEGA_STAR_UCB_: 8,
}


def run_async(function, args_list, njobs, sleep_time_s=0.05):
    with Pool(njobs) as pool:
        with tqdm(total=len(args_list)) as pbar:
            results = {i: pool.apply_async(function, args=args)
                       for i, args in enumerate(args_list)}
            previous_state = 0
            while not all(future.ready() for future in results.values()):
                this_state = np.sum([future.ready() for future in results.values()])
                pbar.update(this_state - previous_state)
                previous_state = this_state
                sleep(sleep_time_s)
            results = [results[i].get() for i in range(len(results))]
    return results



def create_palette(df: pd.DataFrame):
    approaches = np.unique(df[APPROACH])
    return get_palette_for_approaches(approaches)


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def extract_rho(s: str):
    if "rho" not in s:
        return np.nan
    relevant_part = s.split("=")[-1].split("$")[0]
    if r"\frac" in relevant_part:
        relevant_part = relevant_part.split(r"\frac")[-1]
        relevant_part = relevant_part[1:-1]
        numerator, denominator = relevant_part.split("}{")
        numerator = float(numerator)
        denominator = float(denominator)
        return numerator / denominator
    else:
        return float(relevant_part)


def incremental_regret(rew_this, cost_this, rew_best, cost_best):
    return cost_this * (rew_best / cost_best - rew_this / cost_this)


def save_df(df: pd.DataFrame, name: str):
    this_dir = pathlib.Path(__file__).parent.resolve()
    fp = os.path.join(this_dir, "results", name + ".parquet")
    df.to_parquet(fp, index=False)


def load_df(name: str):
    this_dir = pathlib.Path(__file__).parent.resolve()
    parquet_dir = os.path.join(this_dir, "results", name + ".parquet")
    if os.path.exists(parquet_dir):
        return pd.read_parquet(parquet_dir)
    else:
        fp = os.path.join(this_dir, "results", name + ".csv")
        df = pd.read_csv(fp)
        df.to_parquet(parquet_dir, index=False)
        return df


def normalize_regret(df: pd.DataFrame):
    df.loc[:, NORMALIZED_REGRET] = np.nan
    for _, gdf in df.groupby([APPROACH, REP, K, MINIMUM_AVERAGE_COST]):
        budget = np.max(gdf[SPENT_BUDGET])
        achievable_reward = budget * gdf[OPTIMAL_REWARD] / gdf[OPTIMAL_COST]
        df.loc[gdf.index, NORMALIZED_REGRET] = (gdf[REGRET] / achievable_reward)
    return df


def normalize_budget(df: pd.DataFrame):
    df = df.sort_values(by=EXPECTED_SPENT_BUDGET)
    for _, gdf in df.groupby([APPROACH, REP, K, MINIMUM_AVERAGE_COST]):
        spent_budget = gdf[EXPECTED_SPENT_BUDGET]
        budget = np.max(gdf[EXPECTED_SPENT_BUDGET])
        expected_spent_budget = spent_budget / np.max(spent_budget) * budget
        normalized = expected_spent_budget / budget
        df.loc[gdf.index, NORMALIZED_BUDGET] = normalized
    return df


def add_zero(df: pd.DataFrame):
    rows = []
    for _, gdf in df.groupby([APPROACH, K, REP]):
        row = gdf.iloc[0]
        row[NORMALIZED_BUDGET] = 0
        row[SPENT_BUDGET] = 0
        row[REGRET] = 0
        row[NORMALIZED_REGRET] = 0
        rows.append(row)
    additional_rows = pd.DataFrame(rows, columns=df.columns).reset_index()
    return pd.concat([additional_rows, df]).reset_index()


def adjust_approach_names(df: pd.DataFrame):
    assert not np.any(np.isnan(df.loc[:, RHO]))
    df[APPROACH] = df[RHO].apply(
        lambda x: OMEGA_UCB_STUMP.format(round(x, 2) if x < 1 else int(x))
    )
    return df


def prepare_df(df: pd.DataFrame, n_steps=10):
    df.ffill(inplace=True)
    df = normalize_budget(df)
    df = normalize_regret(df)
    df.sort_values(by=[NORMALIZED_BUDGET, APPROACH], inplace=True)
    df.loc[:, NORMALIZED_BUDGET] = np.ceil(df[NORMALIZED_BUDGET] * 10) / 10
    df = df[df[NORMALIZED_BUDGET] <= 1]
    df = df.groupby([K, APPROACH, NORMALIZED_BUDGET, REP, MINIMUM_AVERAGE_COST]).last().reset_index()
    df.loc[:, RHO] = np.nan
    df.loc[:, RHO] = df[APPROACH].apply(lambda x: extract_rho(x))
    df.loc[:, IS_OUR_APPROACH] = False
    df.loc[:, IS_OUR_APPROACH] = df[APPROACH].apply(lambda x: OMEGA_UCB_ in x)
    df.loc[:, APPROACH_ORDER] = np.nan
    df = add_zero(df)
    df[K] = df[K].astype(int)
    return df


def create_custom_legend(grid: sns.FacetGrid, with_markers: bool = False, with_dashes: bool = True):
    app_color_list = color_list()
    approaches = [entry[0] for entry in app_color_list]
    marker_dict = get_markers_for_approaches(approaches)
    dash_dict = get_linestyles_for_approaches(approaches)
    colors = [entry[1] for entry in app_color_list]
    if with_markers:
        custom_lines = [Line2D([0], [0], color=color, marker=marker, markeredgewidth=0.2)
                        for color, marker in zip(colors, marker_dict.values())]
    elif with_dashes:
        custom_lines = [Line2D([0], [0], color=color, linestyle=(0, dash), markeredgewidth=0.2)
                        for color, dash in zip(colors, dash_dict.values())]
    else:
        custom_lines = [Line2D([0], [0], color=color)
                        for color, marker in zip(colors, marker_dict.values())]
    axes = grid.axes
    for ax in axes.flatten():
        if ax.legend():
            ax.legend().remove()
    plt.figlegend(custom_lines, approaches,
                  bbox_to_anchor=(0, 0.68, 1, 0.2),
                  loc="lower left",
                  mode="expand",
                  borderaxespad=1,
                  ncol=6
                  )


def create_multinomial_parameters(rng, k):
    params = rng.uniform(size=(k, 5))
    sum_params = np.sum(params, axis=1)
    assert len(sum_params) == k
    return params / np.expand_dims(sum_params, -1)


def get_average_multinomial(params: np.ndarray):
    indexes = np.array([
        [0, 1, 2, 3, 4]
        for _ in range(len(params))
    ])
    assert indexes.shape == params.shape
    weighted_indexes = params * indexes
    mean = np.mean(weighted_indexes, axis=1)
    assert len(mean) == len(params)
    return mean


def concat_dataframes():
    data_path = os.path.join(os.getcwd(), "results")
    kdd_submission_path = os.path.join(data_path, "results_kdd_submission")
    for file in os.listdir(kdd_submission_path):
        name = os.path.splitext(file)[0]
        kdd_df = pd.read_parquet(os.path.join(kdd_submission_path, file))
        ucbb2_df = pd.read_parquet(os.path.join(data_path, name + "_ucbb2.parquet"))
        df = pd.concat([kdd_df, ucbb2_df], ignore_index=True)
        df.to_parquet(os.path.join(data_path, name + ".parquet"))


if __name__ == '__main__':
    concat_dataframes()
