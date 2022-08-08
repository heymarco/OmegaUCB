from typing import Tuple

import numpy as np

from .abstract import Strategy, Oracle, QueryStrategy, Dataset, LabelCorrector
from experiment_logging import logger


class StandardStrategy(Strategy):
    def __init__(self,
                 name: str,
                 dataset: Dataset,
                 query_strategy: QueryStrategy,
                 oracle: Oracle,
                 corrector: LabelCorrector,
                 learner,
                 test_data: Tuple[np.ndarray, np.ndarray]):
        self.name = name
        self.dataset = dataset
        self.query_strategy = query_strategy
        self.oracle = oracle
        self.corrector = corrector
        self.learner = learner
        self.test_data = test_data
        self._n_queries = 0

    def execute_round(self):
        # 1. get unlabeled pool of data
        D, d_indices = self.dataset.D()
        # 2. execute query (instances, indices)
        query_index = self.query_strategy.query(D, d_indices)
        # 3. get instance for query index (index)
        queried_instance = D[query_index]
        # 4. ask oracle for label (instance, index) -> gives flexibility
        if not self.dataset.has_label_for_instance(query_index):
            this_label = self.oracle.get_noisy_label(queried_instance, query_index)
            # 5. add label to labeled pool (label, index)
            self.dataset.digest(query_index, this_label)
            self._n_queries += 1
        else:
            this_label = self.dataset.label_for_instance(query_index)

        if len(self.dataset) < 10:
            return

        # 6. check label
        L, labels, _ = self.dataset.L(exclude_index=query_index)
        should_relabel = self.corrector.should_relabel(L, labels, queried_instance, this_label)
        _, confusion = self.corrector.is_confused(queried_instance, this_label)
        if should_relabel:
            new_label = self.oracle.get_noisy_label(queried_instance, query_index)
            self.dataset.digest(query_index, new_label)
            self._n_queries += 1
            while np.isnan(self.dataset.label_for_instance(query_index)):
                new_label = self.oracle.get_noisy_label(queried_instance, query_index)
                self.dataset.digest(query_index, new_label)
                self._n_queries += 1
        this_label = self.dataset.label_for_instance(query_index)
        # 7. evaluate learner
        L, labels, _ = self.dataset.L()
        self.learner.fit(L, labels)
        acc = self.learner.score(self.test_data[0], self.test_data[1])

        # 8. log progress
        logger.track_confusion(confusion)
        logger.track_corrector(self.corrector.name)
        logger.track_strategy(self.name)
        logger.track_accuracy(acc)
        logger.track_n_data(len(labels))
        logger.track_should_relabel(should_relabel)
        logger.track_corrected_label(this_label)
        logger.track_true_label(self.oracle.get_clean_label(query_index))

    def num_queries(self) -> int:
        return self._n_queries
