from typing import List

import numpy as np
import pandas as pd


class ExperimentLogger:

    def __init__(self,
                 columns: List = ["strategy", "corrector", "index", "n-data", "confusion", "is-correct",
                                  "corrected-label", "true-label", "instance-index", "should-relabel",
                                  "accuracy"]):
        self._data = []
        self._columns = columns
        self._current_row = [np.nan for _ in range(len(self._columns))]

    def track_strategy(self, newval):
        self._track_value(newval, "strategy")

    def track_corrector(self, newval):
        self._track_value(newval, "corrector")

    def track_index(self, newval):
        self._track_value(newval, "index")

    def track_n_data(self, newval):
        self._track_value(newval, "n-data")

    def track_confusion(self, newval):
        self._track_value(newval, "confusion")

    def track_is_correct(self, newval):
        self._track_value(newval, "is-correct")

    def track_corrected_label(self, newval):
        self._track_value(newval, "corrected-label")

    def track_true_label(self, newval):
        self._track_value(newval, "true-label")

    def track_instance_index(self, newval):
        self._track_value(newval, "instance-index")

    def track_accuracy(self, newval):
        self._track_value(newval, "accuracy")

    def track_should_relabel(self, newval):
        self._track_value(newval, "should-relabel")

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
