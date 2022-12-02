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
        "bandit_comparison_ci_rgeq1.csv",
        "bandit_comparison_ci_rleq1.csv",
        "bandit_comparison_ucbsc.csv"
    ]
    final_df_name = "bandit_comparison_combined.csv"
    df = combine_dataframes(directory, filenames)
    assert not os.path.exists(os.path.join(directory, final_df_name))
    df.to_csv(os.path.join(directory, final_df_name), index=False)
