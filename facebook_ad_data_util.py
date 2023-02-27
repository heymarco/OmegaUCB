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
    high_revenue_outliers = filtered_df["revenue_per_1000_impressions"] >= 1
    filtered_df = filtered_df.loc[np.invert(high_revenue_outliers)]
    return filtered_df


def get_setting(df):
    mean_rewards = np.array(df["revenue_per_1000_impressions"])
    mean_costs = np.array(df["cost_per_1000_impressions"])
    mean_costs = mean_costs[mean_costs > 0]
    mean_rewards = mean_rewards[mean_costs > 0]
    return mean_rewards, mean_costs


def add_noise(setting, rng):
    rew = setting[0]
    cost = setting[1]
    noise_rew = rng.uniform(-1, 1, size=rew.shape) * 0.05
    noise_cost = rng.uniform(-1, 1, size=cost.shape) * 0.05
    rew = rew + noise_rew
    cost = cost + noise_cost
    rew = np.maximum(rew, 0.0)
    rew = np.minimum(rew, 1.0)
    cost = np.maximum(cost, 0.01)
    cost = np.minimum(cost, 1.0)
    return rew, cost


def sort_setting(setting):
    rew = setting[0]
    cost = setting[1]
    efficiency_inv = [c / r if r > 0 else np.infty for c, r in zip(cost, rew)]
    argsort = np.argsort(efficiency_inv)
    return rew[argsort], cost[argsort]


def get_facebook_ad_data_settings(rng):
    data = prepare_facebook_data()
    settings = []
    for _, gdf in data.groupby(["campaign_id", "age", "gender"]):
        setting = get_setting(gdf)
        setting = sort_setting(setting)
        mask = setting[0] > 0
        setting = setting[0][mask], setting[1][mask]
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