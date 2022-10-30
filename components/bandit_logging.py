import time
import uuid

import pandas as pd
import numpy as np


class BanditLogger:

    def __init__(self,
                 columns: list = ["rep", "approach", "k", "high-variance", "optimal-reward", "spent-budget",
                                  "optimal-cost", "reward", "cost", "arm", "p-min"]):
        self._data = []
        self._columns = columns
        self._current_row = [np.nan for _ in range(len(self._columns))]

    def track_approach(self, app: str):
        self._track_value(app, "approach")

    def track_high_variance(self, hv: bool):
        self._track_value(hv, "high-variance")

    def track_p_min(self, p):
        self._track_value(p, "p-min")

    def track_arm(self, arm: int):
        self._track_value(arm, "arm")

    def track_k(self, k: int):
        self._track_value(k, "k")

    def track_rep(self, rep: int):
        self._track_value(rep, "rep")

    def track_optimal_reward(self, value: float):
        self._track_value(value, "optimal-reward")

    def track_optimal_cost(self, value: float):
        self._track_value(value, "optimal-cost")

    def track_reward(self, value: float):
        self._track_value(value, "reward")

    def track_cost(self, value: float):
        self._track_value(value, "cost")

    def track_spent_budget(self, value: float):
        self._track_value(value, "spent-budget")

    def finalize_round(self):
        self._data.append(self._current_row)
        self._current_row = [np.nan for _ in range(len(self._columns))]

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._data, columns=self._columns)
        self._reset()
        return df

    def _track_value(self, newval, id):
        self._current_row[self._index_of(id)] = newval

    def _index_of(self, id):
        index = next(i for i in range(len(self._columns)) if self._columns[i] == id)
        return index

    def _reset(self):
        self._current_row = [np.nan for _ in range(len(self._columns))]
        self._data = []


logger = BanditLogger()
