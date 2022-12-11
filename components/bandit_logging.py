import time

import pandas as pd
import numpy as np


REP = "rep"
ROUND = r"$t$"
BEST_ARM = "best-arm"
APPROACH = "Approach"
TIME = "time"
K = r"$K$"
CURRENT_ARM = r"$I_t$"
OPTIMAL_REWARD = r"$r_1$"
OPTIMAL_COST = r"$c_1$"
SPENT_BUDGET = r"spent-budget"
TOTAL_REWARD = "reward"
AVG_COST_OF_CURRENT_ARM = r"$\mu_i^c$"
AVG_REWARD_OF_CURRENT_ARM = r"$\mu_i^r$"
COST_OF_CURRENT_ARM = r"$c_{i,t}$"
REWARD_OF_CURRENT_ARM = r"$r_{i,t}$"
MINIMUM_AVERAGE_COST = r"$c_{min}$"
REGRET = "Regret"
NORMALIZED_BUDGET = "Normalized Budget"
RHO = r"$\rho$"
IS_OUR_APPROACH = "our_approach"
APPROACH_ORDER = "order"
NORMALIZED_REGRET = "Normalized Regret"


all_ids = [
    REP,
    ROUND,
    APPROACH,
    BEST_ARM,
    K,
    CURRENT_ARM,
    OPTIMAL_REWARD,
    OPTIMAL_COST,
    SPENT_BUDGET,
    TOTAL_REWARD,
    AVG_COST_OF_CURRENT_ARM,
    AVG_REWARD_OF_CURRENT_ARM,
    COST_OF_CURRENT_ARM,
    REWARD_OF_CURRENT_ARM,
    MINIMUM_AVERAGE_COST,
    REGRET,
    TIME,
    NORMALIZED_BUDGET,
    APPROACH_ORDER
]


class BanditLogger:

    def __init__(self):
        self._columns = all_ids
        self._data = []
        self._current_row = [np.nan for _ in range(len(self._columns))]
        self._column_indices = {key: i for (i, key) in enumerate(self._columns)}

    def track_approach(self, value: str):
        self._track_value(value, APPROACH)

    def track_round(self, value: int):
        self._track_value(value, ROUND)

    def track_regret(self, value: float):
        self._track_value(value, REGRET)

    def track_c_min(self, value: float):
        self._track_value(value, MINIMUM_AVERAGE_COST)

    def track_normalized_budget(self, value: float):
        self._track_value(value, NORMALIZED_BUDGET)

    def track_arm(self, value: int):
        self._track_value(value, CURRENT_ARM)

    def track_best_arm(self, value: int):
        self._track_value(value, BEST_ARM)

    def track_total_reward(self, value: int):
        self._track_value(value, TOTAL_REWARD)

    def track_k(self, value: int):
        self._track_value(value, K)

    def track_rep(self, value: int):
        self._track_value(value, REP)

    def track_optimal_reward(self, value: float):
        self._track_value(value, OPTIMAL_REWARD)

    def track_optimal_cost(self, value: float):
        self._track_value(value, OPTIMAL_COST)

    def track_reward_sample(self, value: float):
        self._track_value(value, REWARD_OF_CURRENT_ARM)

    def track_cost_sample(self, value: float):
        self._track_value(value, COST_OF_CURRENT_ARM)

    def track_spent_budget(self, value: float):
        self._track_value(value, SPENT_BUDGET)

    def track_mean_rew_current_arm(self, value: float):
        self._track_value(value, AVG_REWARD_OF_CURRENT_ARM)

    def track_mean_cost_current_arm(self, value: float):
        self._track_value(value, AVG_COST_OF_CURRENT_ARM)

    def finalize_round(self):
        self._data.append(self._current_row)
        self._current_row = [np.nan for _ in range(len(self._columns))]

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._data, columns=self._columns)
        self._reset()
        return df

    def track_time(self):
        current_time = time.perf_counter_ns()
        self._track_value(current_time, "time")

    def _track_value(self, newval, id):
        self._current_row[self._index_of(id)] = newval

    def _index_of(self, id):
        return self._column_indices[id]

    def _reset(self):
        self._current_row = [np.nan for _ in range(len(self._columns))]
        self._data = []


logger = BanditLogger()
