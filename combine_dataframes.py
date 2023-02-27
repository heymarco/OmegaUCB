import os.path

import pandas as pd


def combine_dataframes(directory: str, filenames: list):
    dfs = [
        pd.read_parquet(os.path.join(directory, fn)) for fn in filenames
    ]
    return pd.concat(dfs, ignore_index=True)


if __name__ == '__main__':
    directory = os.path.join(os.getcwd(), "results")
    filenames = [
        "synth_bernoulli.parquet",
        "synth_bernoulli_rest.parquet"
    ]
    final_df_name = "synth_bernoulli_combined.parquet"
    df = combine_dataframes(directory, filenames)
    assert not os.path.exists(os.path.join(directory, final_df_name))
    df.to_parquet(os.path.join(directory, final_df_name), index=False)
