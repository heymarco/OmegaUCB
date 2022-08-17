import time
import uuid

import pandas as pd
import numpy as np


class ExperimentLogger:

    def __init__(self,
                 columns: list = ["rep", "approach", "parameters", "dataset", "time", "noise-level", "alpha", "beta",
                                  "estimated-score", "true-score", "t", "n", "gradients", "vectors"]):
        self._data = []
        self._columns = columns
        self._current_row = [np.nan for _ in range(len(self._columns))]

    def track_rep(self, rep: int):
        self._track_value(rep, "rep")

    def track_dataset_name(self, name):
        self._track_value(name, "dataset")

    def track_time(self):
        current_time = time.perf_counter_ns()
        self._track_value(current_time, "time")

    def track_estimated_score(self, score):
        self._track_value(score, "estimated-score")

    def track_true_score(self, score):
        self._track_value(score, "true-score")

    def track_t_n(self, t, n):
        self._track_value(t, "t")
        self._track_value(n, "n")

    def track_vectors(self, vectors):
        vectors = str(vectors)
        self._track_value(vectors, "vectors")

    def track_gradients(self, gradients):
        gradients = str(gradients)
        self._track_value(gradients, "gradients")

    def track_noise_level(self, noise):
        self._track_value(noise, "noise-level")

    def track_alpha_beta(self, alpha, beta):
        self._track_value(str(alpha), "alpha")
        self._track_value(str(beta), "beta")

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


logger = ExperimentLogger()
