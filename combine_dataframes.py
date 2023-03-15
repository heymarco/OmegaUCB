import os.path
from typing import List

import pandas as pd
from approach_names import *
from components.bandit_logging import APPROACH


def combine_dataframes(directory: str, filenames: list):
    dfs = [
        pd.read_parquet(os.path.join(directory, fn)) for fn in filenames
    ]
    return pd.concat(dfs, ignore_index=True)


def combine_dataframes_exclude_approaches(direcory: str, filenames: list,
                                          exclude: List[List[str]]):
    dfs = []
    for filename, apps in zip(filenames, exclude):
        df = pd.read_parquet(os.path.join(direcory, filename))
        df[APPROACH].ffill(inplace=True)
        for approach in apps:
            df = df[df[APPROACH] != approach]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


if __name__ == '__main__':
    directory = os.path.join(os.getcwd(), "results")
    final_df_name = "synth_beta_combined.parquet"
    filenames = [
        "synth_beta.parquet",
        "synth_beta_30.parquet"
    ]
    exclude = [
        [
            ETA_UCB_1_64,
            ETA_UCB_1_32,
            ETA_UCB_1_16,
            ETA_UCB_1_8,
            ETA_UCB_1_4,
            ETA_UCB_1_2,
            ETA_UCB_1,
            ETA_UCB_2
        ],
        [
            MUCB,
            CUCB,
            IUCB,
            BTS,
            UCB_SC,
            UCB_SC_PLUS,
            BUDGET_UCB,
            B_GREEDY,
            OMEGA_UCB_1_64,
            OMEGA_UCB_1_32,
            OMEGA_UCB_1_16,
            OMEGA_UCB_1_8,
            OMEGA_UCB_1_4,
            OMEGA_UCB_1_2,
            OMEGA_UCB_1,
            OMEGA_UCB_2,
        ]
    ]
    if not exclude:
        df = combine_dataframes(directory, filenames)
    else:
        df = combine_dataframes_exclude_approaches(directory,
                                                   filenames,
                                                   exclude)
    assert not os.path.exists(os.path.join(directory, final_df_name))
    df.to_parquet(os.path.join(directory, final_df_name), index=False)
