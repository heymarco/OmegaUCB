import os.path

import pandas as pd


def combine_dataframes(directory: str, filenames: list):
    dfs = [
        pd.read_csv(os.path.join(directory, fn)) for fn in filenames
    ]
    return pd.concat(dfs, ignore_index=True)


if __name__ == '__main__':
    directory = os.path.join(os.getcwd(), "results")
    filenames = [
        "bandit_comparison.csv",
        "bandit_comparison_2.csv",
    ]
    final_df_name = "bandit_comparison_combined"
    df = combine_dataframes(directory, filenames)
    assert not os.path.exists(os.path.join(directory, final_df_name))
    df.to_csv(os.path.join(directory, final_df_name), index=False)
