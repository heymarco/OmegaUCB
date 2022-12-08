import os.path
import gc
import pathlib
from multiprocessing import Pool
from time import sleep

import numpy as np
import pandas
import pandas as pd
from scipy import stats
from tqdm import tqdm
import seaborn as sns

from components.bandit_logging import *


approach_order = {
    "BTS": 0,
    "UCB-SC+": 1,
    "Budget-UCB": 2,
    "c-UCB": 3,
    "i-UCB": 4,
    "m-UCB": 5,
    "w-UCB": 6
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
    df.sort_values(by=[APPROACH_ORDER, RHO])
    ts = ["BTS"]
    wucb = ["w-UCB"]
    ucb = ["m-UCB", "c-UCB", "i-UCB", "Budget-UCB"]
    ucb_sc = ["UCB-SC"]
    id_list = [ts, ucb_sc, ucb, wucb]
    color_palettes = ["Wistia", "Reds", "Greens", "Blues"]
    final_palette = []
    for c, ids in zip(color_palettes, id_list):
        mask = df[APPROACH].apply(lambda x: np.any([id in x for id in ids])).astype(bool)
        data = df[APPROACH].loc[mask]
        n_colors = len(np.unique(data))
        palette = sns.color_palette(c, n_colors=n_colors + 1)[1:]
        final_palette += palette
    return final_palette


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def extract_rho(s: str):
    if "rho=" not in s:
        return np.nan
    relevant_part = s.split("=")[-1][:-1]
    if "/" in relevant_part:
        numerator, denominator = relevant_part.split("/")
        numerator = float(numerator)
        denominator = float(denominator)
        return numerator / denominator
    else:
        return float(relevant_part)


def incremental_regret(rew_this, cost_this, rew_best, cost_best):
    return cost_this * (rew_best / cost_best - rew_this / cost_this)


def save_df(df: pd.DataFrame, name: str):
    this_dir = pathlib.Path(__file__).parent.resolve()
    fp = os.path.join(this_dir, "results", name + ".csv")
    df.to_csv(fp, index=False)


def load_df(name: str):
    this_dir = pathlib.Path(__file__).parent.resolve()
    fp = os.path.join(this_dir, "results", name + ".csv")
    return pd.read_csv(fp)


def prepare_df2(df: pd.DataFrame):
    df.ffill(inplace=True)
    df.sort_values(by=APPROACH, inplace=True)
    df[NORMALIZED_BUDGET] = np.ceil((df[NORMALIZED_BUDGET] * 10)) / 10
    df = df[df[NORMALIZED_BUDGET] <= 1]
    df[RHO] = np.nan
    df[RHO] = df[APPROACH].apply(lambda x: extract_rho(x))
    df[IS_OUR_APPROACH] = False
    df[IS_OUR_APPROACH] = df[APPROACH].apply(lambda x: "w-UCB" in x)
    df[APPROACH_ORDER] = np.nan
    df[APPROACH_ORDER] = df[APPROACH].apply(
        lambda x: next(approach_order[key] for key in approach_order.keys() if key in x)
    )
    assert not np.any(np.isnan(df[APPROACH_ORDER]))
    return df