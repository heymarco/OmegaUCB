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
    path, ext = os.path.splitext(csv_path)
    newpath = path + "_reduced" + ext
    with pd.read_csv(csv_path, chunksize=int(10e7), low_memory=False, dtype={"rep": float, "approach": str, "k": float,"high-variance": float, "optimal-reward": float, "spent-budget": float, "optimal-cost": float, "reward": float, "cost": float, "arm": float}) as iterator:
        last_row = None
        for chunk_number, chunk in tqdm(enumerate(iterator)):
            if not chunk_number == 0:
                chunk = pd.concat([last_row, chunk]).reset_index()
            chunk.iloc[every_nth::] = chunk.iloc[every_nth::].ffill()
            last_row = chunk.iloc[-1]
            reduced_chunk = chunk.iloc[every_nth::]
            mode = "w" if chunk_number == 0 else "a"
            header = chunk_number == 0
            reduced_chunk.to_csv(newpath, mode=mode, index=False, header=header)
