import os
import pathlib

import numpy as np
import pandas as pd


def load_facebook_data():
    this_dir = pathlib.Path(__file__).parent.resolve()
    fp = os.path.join(this_dir, "data", "KAG_conversion_adapted.csv")
    return pd.read_csv(fp)


def prepare_facebook_data():
    raw_data = load_facebook_data()
    is_zero_cost = raw_data["spent"] == 0
    is_zero_reward = raw_data["approved_conversion"] == 0
    is_nan_ratio = np.isnan(raw_data["reward_cost_ratio"])
    non_informative_rows = np.logical_and(is_zero_reward, is_zero_cost)  # do not include ads for which we have no data
    corrupted_rows = np.logical_and(np.invert(is_zero_reward), is_zero_cost)  # cost although no clicks occurred
    mask = np.invert(np.logical_or(non_informative_rows, corrupted_rows))
    mask = np.logical_or(mask, np.invert(is_nan_ratio))
    filtered_df = raw_data.loc[mask].reset_index()
    high_ratio_outliers = filtered_df["reward_cost_ratio"] > 1
    filtered_df = filtered_df.loc[np.invert(high_ratio_outliers)]
    high_revenue_outliers = filtered_df["revenue_per_1000_impressions"] > 1
    filtered_df = filtered_df.loc[np.invert(high_revenue_outliers)]
    return filtered_df


def sort_df(df):
    df = df.sort_values(by="reward_cost_ratio", ascending=False)
    return df


def get_setting(df):
    df = sort_df(df)
    mean_rewards = list(df["revenue_per_1000_impressions"])
    mean_costs = list(df["cost_per_1000_impressions"])
    return np.array(mean_rewards), np.array(mean_costs)


def get_facebook_ad_data_settings():
    data = prepare_facebook_data()
    settings = []
    for _, gdf in data.groupby(["campaign_id", "age", "gender"]):
        setting = get_setting(gdf)
        k = len(setting[0])
        if k >= 2:
            settings.append(get_setting(gdf))
    return settings


def get_facebook_ad_stats():
    data = prepare_facebook_data()
    cols = ["Age", "Gender", "c_min", "K"]
    rows = []
    for (cid, age, gender), gdf in data.groupby(["campaign_id", "age", "gender"]):
        setting = get_setting(gdf)
        k = len(setting[0])
        if k >= 2:
            rows.append([
                age, gender, min(setting[-1]), len(setting[0])
            ])
    df = pd.DataFrame(rows, columns=cols)
    print(df)