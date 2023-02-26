import os.path
import gc
import pathlib
from multiprocessing import Pool
from time import sleep

import numpy as np
import pandas
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from tqdm import tqdm
import seaborn as sns

from colors import color_list, get_palette_for_approaches
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
    ETA_UCB_: 8,
}


def str_to_arr(s, dtype):
    return np.fromstring(s[1:-1], dtype=dtype, sep=" ")


def linear_cost_function(t, c0=1, k=1 / 5, t0=1) -> float:
    return c0 + k * (t - t0)


def run_async(function, args_list, njobs, sleep_time_s=0.05):
    pool = Pool(njobs)
    results = {i: pool.apply_async(function, args=args)
               for i, args in enumerate(args_list)}
    while not all(future.ready() for future in results.values()):
        sleep(sleep_time_s)
    results = [results[i].get() for i in range(len(results))]
    pool.close()
    return results


def reg_beta(x, a, b, k=100):
    n = a + b
    reg = k / n ** 2
    return stats.beta.pdf(x, a + reg, b + reg)


def subsample_csv(csv_path: str, every_nth: int = 1):
    path, ext = os.path.splitext(csv_path)
    newpath = path + "_reduced" + ext
    reduced_chunks = []
    with pd.read_csv(csv_path, chunksize=1e7, low_memory=False, dtype={"rep": float, "approach": str, "k": float,"high-variance": float, "optimal-reward": float, "spent-budget": float, "optimal-cost": float, "reward": float, "cost": float, "arm": float}) as iterator:
        for chunk_number, chunk in tqdm(enumerate(iterator)):
            necessary_rows = np.logical_or(np.invert(np.isnan(chunk["rep"])), chunk["approach"].apply(lambda x: str(x) != "nan"))
            necessary_rows = np.logical_or(necessary_rows, np.invert(np.isnan(chunk["high-variance"])))
            necessary_rows = np.logical_or(necessary_rows, np.invert(np.isnan(chunk["k"])))
            necessary_rows = chunk.loc[necessary_rows.astype(bool)]
            reduced_chunk = chunk.iloc[::every_nth]
            necessary_rows = necessary_rows.index
            selected_rows = reduced_chunk.index
            selected_rows = np.unique(np.concatenate([necessary_rows, selected_rows]))
            final_chunk = chunk.loc[selected_rows]
            reduced_chunks.append(final_chunk)
    df = pd.concat(reduced_chunks).reset_index()
    df.to_csv(newpath, index=False)


def create_palette(df: pd.DataFrame):
    approaches = np.unique(df[APPROACH])
    return get_palette_for_approaches(approaches)


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


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
    for _, gdf in df.groupby([APPROACH, REP, K]):
        budget = np.max(gdf[SPENT_BUDGET])
        achievable_reward = budget * gdf[OPTIMAL_REWARD] / gdf[OPTIMAL_COST]
        df.loc[gdf.index, NORMALIZED_REGRET] = (gdf[REGRET] / achievable_reward)
    return df

def normalize_budget(df: pd.DataFrame):
    df = df.sort_values(by=EXPECTED_SPENT_BUDGET)
    for _, gdf in df.groupby([APPROACH, REP, K]):
        expected_spent_budget = gdf[EXPECTED_SPENT_BUDGET]
        budget = np.max(gdf[SPENT_BUDGET])
        normalized = expected_spent_budget / budget
        df.loc[gdf.index, NORMALIZED_BUDGET] = normalized
    return df

def remove_outliers(df: pd.DataFrame):
    for _, gdf in df.groupby([APPROACH, K, NORMALIZED_BUDGET]):
        min_percentile = np.percentile(gdf[NORMALIZED_REGRET], q=2)
        max_percentile = np.percentile(gdf[NORMALIZED_REGRET], q=98)
        mask = np.logical_or(gdf[NORMALIZED_REGRET] <= min_percentile, gdf[NORMALIZED_REGRET] >= max_percentile)
        nan_indices = gdf.index[mask]
        df.loc[nan_indices, NORMALIZED_REGRET] = np.nan
    return df


def adjust_approach_names(df: pd.DataFrame):
    assert not np.any(np.isnan(df.loc[:, RHO]))
    df[APPROACH] = df[RHO].apply(
        lambda x: OMEGA_UCB_STUMP.format(round(x, 2) if x < 1 else int(x))
    )
    return df


def prepare_df(df: pd.DataFrame, n_steps=10):
    df.ffill(inplace=True)
    df = df[df[ROUND] <= 1.5e5]
    df = normalize_budget(df)
    df = normalize_regret(df)
    df.sort_values(by=APPROACH, inplace=True)
    df.loc[:, NORMALIZED_BUDGET] = np.ceil((df[NORMALIZED_BUDGET] * n_steps)) / (n_steps)
    df = df[df[NORMALIZED_BUDGET] <= 1]
    df = df.groupby([K, APPROACH, NORMALIZED_BUDGET, REP]).max().reset_index()
    df = remove_outliers(df)
    df.loc[:, RHO] = np.nan
    df.loc[:, RHO] = df[APPROACH].apply(lambda x: extract_rho(x))
    df.loc[:, IS_OUR_APPROACH] = False
    df.loc[:, IS_OUR_APPROACH] = df[APPROACH].apply(lambda x: OMEGA_UCB_ in x)
    df.loc[:, APPROACH_ORDER] = np.nan
    df[APPROACH_ORDER] = df[APPROACH].apply(
        lambda x: next(approach_order[key] for key in approach_order.keys() if key in x)
    )
    lens = []
    dists = []
    for _, gdf in df[np.logical_and(df[APPROACH] == ETA_UCB_1_4, df[K] == 50)].groupby([NORMALIZED_BUDGET]):
        lens.append(len(gdf))
        dists.append(gdf[NORMALIZED_REGRET])
    print(lens)
    print(dists)
    assert not np.any(np.isnan(df[APPROACH_ORDER]))
    return df


def move_legend_below_graph(grid, ncol: int, title: str):
    sns.move_legend(grid, "lower center", ncols=ncol, title=title)
    plt.tight_layout()


def create_custom_legend(grid: sns.FacetGrid):
    app_color_list = color_list()
    approaches = [entry[0] for entry in app_color_list]
    colors = [entry[1] for entry in app_color_list]

    custom_lines = [Line2D([0], [0], color=color, lw=2) for color in colors]

    axes = grid.axes
    for ax in axes.flatten():
        if ax.legend():
            ax.legend().remove()
    plt.figlegend(custom_lines, approaches,
                  bbox_to_anchor=(0, 0.72, 1, 0.2),
                  loc="lower left",
                  mode="expand",
                  borderaxespad=1,
                  ncol=6
                  )
