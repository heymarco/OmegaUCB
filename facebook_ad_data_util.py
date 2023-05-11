import os
import pathlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_facebook_data():
    this_dir = pathlib.Path(__file__).parent.resolve()
    fp = os.path.join(this_dir, "data", "processed.csv")
    if not os.path.exists(fp):
        prepare_raw_data()
        load_facebook_data()
    return pd.read_csv(fp)


def prepare_raw_data():
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


def prepare_facebook_data():
    raw_data = load_facebook_data()
    is_zero_cost = raw_data["spent"] == 0
    has_no_clicks = raw_data["clicks"] == 0
    is_zero_reward = raw_data["total_conversion"] == 0
    is_nan_ratio = np.isnan(raw_data["reward_cost_ratio"])
    non_informative_rows = np.logical_and(np.logical_and(is_zero_reward, is_zero_cost), has_no_clicks)  # do not include ads for which we have no data
    corrupted_rows = np.logical_and(np.invert(is_zero_reward), is_zero_cost)  # cost although no clicks occurred
    mask = np.invert(np.logical_or(non_informative_rows, corrupted_rows))
    mask = np.logical_or(mask, np.invert(is_nan_ratio))
    filtered_df = raw_data.loc[mask].reset_index()
    return filtered_df


def get_setting(df):
    mean_rewards = np.array(df["rpc"])
    mean_costs = np.array(df["cpc"])
    mean_costs = mean_costs[mean_costs > 0]
    mean_rewards = mean_rewards[mean_costs > 0]
    return mean_rewards, mean_costs


def scale_randomly(setting, rng):
    rew = setting[0]
    cost = setting[1]
    min_r = rng.uniform(0, 0.5)
    max_r = rng.uniform(0.5, 1)
    min_c = rng.uniform(0, 0.5)
    max_c = rng.uniform(0.5, 1)
    rew = MinMaxScaler(feature_range=(min_r, max_r)).fit_transform(np.expand_dims(rew, -1)).flatten()
    cost = MinMaxScaler(feature_range=(min_c, max_c)).fit_transform(np.expand_dims(cost, -1)).flatten()
    return rew, cost


def sort_setting(setting):
    rew = setting[0]
    cost = setting[1]
    efficiency_inv = [c / r if r > 0 else np.infty for c, r in zip(cost, rew)]
    argsort = np.argsort(efficiency_inv)
    return rew[argsort], cost[argsort]


def normalize_setting(setting):
    rew, cost = setting
    rew[rew == 0] = 0.01
    cost[cost == 0] = 0.01
    norm_rew = rew / np.max(rew) * 0.99
    norm_cost = cost / np.max(cost) * 0.99
    return norm_rew, norm_cost


def get_facebook_ad_data_settings(rng):
    data = prepare_facebook_data()
    settings = []
    for _, gdf in data.groupby(["campaign_id", "age", "gender"]):
        setting = get_setting(gdf)
        setting = normalize_setting(setting)
        setting = sort_setting(setting)
        k = len(setting[0])
        if k >= 2:
            settings.append(setting)
    return settings


def get_facebook_ad_stats():
    data = prepare_facebook_data()
    cols = ["campaign_id", "age", "gender", "c_min", "K"]
    rows = []
    for (cid, age, gender), gdf in data.groupby(["campaign_id", "age", "gender"]):
        setting = get_setting(gdf)
        k = len(setting[0])
        if k >= 2:
            rows.append([
                cid, age, gender, min(setting[-1]), len(setting[0])
            ])
    df = pd.DataFrame(rows, columns=cols)
    print(df.sort_values(by=["age", "gender"]).set_index(["age", "gender"]).to_latex(escape=False, multirow=True))