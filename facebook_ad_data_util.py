import os
import pathlib

import numpy as np
import pandas as pd


def _load_facebook_data():
    """
    Loads the dataset that contains the ad campaigns
    If the processed data set does not yet exist, this method calls _prepare_raw_data(), which creates it.
    :return: the dataframe with the processed facebook advertisement data
    """
    this_dir = pathlib.Path(__file__).parent.resolve()
    fp = os.path.join(this_dir, "data", "processed.csv")
    if not os.path.exists(fp):
        _prepare_raw_data()
        _load_facebook_data()
    return pd.read_csv(fp)


def _prepare_raw_data():
    """
    Extracts the statistics for the advertisement campaigns from the raw data set
    """
    this_dir = pathlib.Path(__file__).parent.resolve()
    fp = os.path.join(this_dir, "data", "data.csv")
    assert os.path.exists(fp), "It seems that the data.csv file is missing. Please follow the instructions in our readme under 'Downloading advertisement data'."
    raw_data = pd.read_csv(fp)
    raw_data = raw_data[raw_data["spent"] > 0]
    spent = raw_data["spent"]
    revenue = raw_data["total_conversion"]
    clicks = raw_data["clicks"]
    cpc = spent / clicks
    rpc = revenue / clicks
    raw_data["cpc"] = cpc
    raw_data["rpc"] = rpc
    raw_data = raw_data[clicks >= revenue]
    raw_data["reward_cost_ratio"] = raw_data["rpc"] / raw_data["cpc"]
    raw_data.to_csv(os.path.join("data", "processed.csv"), index=False)


def _prepare_facebook_data():
    """
    Loads and cleans the facebook data set
    """
    raw_data = _load_facebook_data()
    is_zero_cost = raw_data["spent"] == 0
    has_no_clicks = raw_data["clicks"] == 0
    is_zero_reward = raw_data["total_conversion"] == 0
    is_nan_ratio = np.isnan(raw_data["reward_cost_ratio"])
    non_informative_rows = np.logical_and(np.logical_and(is_zero_reward, is_zero_cost),
                                          has_no_clicks)  # do not include ads for which we have no data
    corrupted_rows = np.logical_and(np.invert(is_zero_reward), is_zero_cost)  # cost although no clicks occurred
    mask = np.invert(np.logical_or(non_informative_rows, corrupted_rows))
    mask = np.logical_or(mask, np.invert(is_nan_ratio))
    filtered_df = raw_data.loc[mask].reset_index()
    return filtered_df


def _get_setting(df):
    """
    Get the arms' mean rewards and costs
    """
    mean_rewards = np.array(df["rpc"])
    mean_costs = np.array(df["cpc"])
    mean_costs = mean_costs[mean_costs > 0]
    mean_rewards = mean_rewards[mean_costs > 0]
    return mean_rewards, mean_costs


def _sort_setting(setting):
    """
    Sort the arms by the reward cost ratio in descending order (s.th. the best arm comes at position 0)
    """
    rew = setting[0]
    cost = setting[1]
    efficiency_inv = [c / r if r > 0 else np.infty for c, r in zip(cost, rew)]
    argsort = np.argsort(efficiency_inv)
    return rew[argsort], cost[argsort]


def _normalize_setting(setting):
    """
    Scale the settings to the range [0.01, 0.99]
    """
    rew, cost = setting
    rew[rew == 0] = 0.01
    cost[cost == 0] = 0.01
    norm_rew = rew / np.max(rew) * 0.99
    norm_cost = cost / np.max(cost) * 0.99
    return norm_rew, norm_cost


def get_facebook_ad_data_settings():
    """
    Returns all ad campaigns with at least two arms.
    The settings are sorted and normalized.
    """
    data = _prepare_facebook_data()
    settings = []
    for _, gdf in data.groupby(["campaign_id", "age", "gender"]):
        setting = _get_setting(gdf)
        setting = _normalize_setting(setting)
        setting = _sort_setting(setting)
        k = len(setting[0])
        if k >= 2:
            settings.append(setting)
    return settings
