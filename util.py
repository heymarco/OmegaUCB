import os.path
import gc
from multiprocessing import Pool
from time import sleep

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

MNIST = 554
MUSHROOM = 24


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
            if chunk_number > 10:
                break
    df = pd.concat(reduced_chunks).reset_index()
    df.to_csv(newpath, index=False)
