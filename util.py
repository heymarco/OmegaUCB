import os.path
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


def run_async(function, args_list, njobs, sleep_time_s = 0.05):
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
    iterator = pd.read_csv(csv_path, chunksize=int(1e6))
    reduced_chunks = []
    last_row = None
    for chunk_number, chunk in tqdm(enumerate(iterator)):
        if not chunk_number == 0:
            chunk = pd.concat([last_row, chunk]).reset_index()
        chunk.ffill()
        last_row = chunk.iloc[-1]
        reduced_chunk = chunk.iloc[::every_nth]
        reduced_chunks.append(reduced_chunk)
    reduced_df = pd.DataFrame([reduced_chunks]).reset_index()
    path, ext = os.path.splitext(csv_path)
    reduced_df.to_csv(os.path.join(path, "_reduced" + ext))
