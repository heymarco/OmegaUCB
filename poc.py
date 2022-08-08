import copy
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, fetch_20newsgroups, fetch_20newsgroups_vectorized, \
    load_digits
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from components.correctors import RandomForestLC, DummyLC, BinaryLC
from components.oracles import SymmetricNoiseOracle
from components.datasets import MajorityVotedDataset, StandardDataset
from components.query_strategies import RandomQueryStrategy
from components.strategies import StandardStrategy
from experiment_logging import logger

if __name__ == '__main__':
    noise_level = 0.2
    confusion_thresh = .8
    num_queries = 100
    reps = 3

    data = load_digits()
    x = data.data
    y = data.target

    result = []

    for seed in range(reps):
        rng = np.random.default_rng(seed)

        data_indices = np.arange(len(x)).tolist()
        rng.shuffle(data_indices)
        train_indices = data_indices[:int(0.5 * len(data_indices))]
        test_indices = data_indices[int(0.5 * len(data_indices)):]
        x_train = x[train_indices]
        y_train = y[train_indices]
        x_test = x[test_indices]
        y_test = y[test_indices]
        train_indices = np.arange(len(y_train))

        mv_dataset = MajorityVotedDataset(data=x_train, seed=seed)
        standard_dataset = StandardDataset(data=x_train, seed=seed)
        query_strategy = RandomQueryStrategy(seed)
        oracle = SymmetricNoiseOracle(true_labels=y_train,
                                      noise_level=noise_level,
                                      seed=seed)
        perfect_oracle = SymmetricNoiseOracle(true_labels=y_train,
                                              noise_level=0.0,
                                              seed=seed)
        rf_corrector = RandomForestLC(confusion_thresh=confusion_thresh,
                                      seed=seed)
        binary_corrector = BinaryLC(num_classes=len(np.unique(y_train)),
                                    proba_thresh=.55,
                                    seed=seed)
        dummy_strategy = StandardStrategy(name="traditional",
                                          dataset=copy.deepcopy(standard_dataset),
                                          query_strategy=copy.deepcopy(query_strategy),
                                          oracle=copy.deepcopy(oracle),
                                          corrector=DummyLC(),
                                          learner=RandomForestClassifier(random_state=seed),
                                          test_data=(x_test, y_test))
        topline_clean = StandardStrategy(name="topline",
                                         dataset=copy.deepcopy(standard_dataset),
                                         query_strategy=copy.deepcopy(query_strategy),
                                         oracle=perfect_oracle,
                                         corrector=DummyLC(),
                                         learner=RandomForestClassifier(random_state=seed),
                                         test_data=(x_test, y_test))
        relabel_strategy = StandardStrategy(name="relabel",
                                            dataset=copy.deepcopy(standard_dataset),
                                            query_strategy=query_strategy,
                                            oracle=oracle,
                                            corrector=rf_corrector,
                                            learner=RandomForestClassifier(random_state=seed),
                                            test_data=(x_test, y_test))
        pseudolabel_strategy = StandardStrategy(name="pseudolabels",
                                            dataset=copy.deepcopy(standard_dataset),
                                            query_strategy=query_strategy,
                                            oracle=oracle,
                                            corrector=binary_corrector,
                                            learner=RandomForestClassifier(random_state=seed),
                                            test_data=(x_test, y_test))

        strategies = [
            pseudolabel_strategy,
            relabel_strategy,
            topline_clean,
            dummy_strategy
        ]

        for strategy in strategies:
            while strategy.num_queries() < num_queries:
                strategy.execute_round()
                logger.track_index(strategy.num_queries())
                logger.finalize_round()
            df = logger.get_dataframe()
            result.append(df)

    result_df = pd.concat(result, axis=0, ignore_index=True)
    result_df.to_csv(os.path.join(os.getcwd(), "results", "poc.csv"), index=False)

    #     x_train_cleaned, y_train_cleaned, queried_indices = dataset.L()
    #
    #     x_train_noisy = x_train[queried_indices]
    #     y_train_noisy = np.array([dataset.noisy_label_for_instance(i)
    #                               for i in queried_indices])
    #     x_train_clean = x_train_noisy
    #     y_train_clean = np.array([oracle.get_clean_label(i)
    #                               for i in queried_indices])
    #
    #     non_queried_boolean = np.zeros_like(y_train)
    #     non_queried_boolean[queried_indices] = 1
    #     non_queried_indices = np.arange(len(y_train))[non_queried_boolean]
    #     additional_queries = np.random.choice(non_queried_indices, num_queries - dataset.n_data())
    #     x_train_large = np.concatenate([x_train_clean,
    #                                     x_train[additional_queries]], axis=0)
    #     y_train_large_clean = np.concatenate([y_train_clean,
    #                                           y_train[additional_queries]])
    #     y_train_large_noisy = [oracle.get_noisy_label(i) for i in additional_queries]
    #     y_train_large_noisy = np.concatenate([y_train_noisy, y_train_large_noisy], axis=0)
    #
    #     model_cleaned_data = RandomForestClassifier(random_state=seed)
    #     model_clean_data = RandomForestClassifier(random_state=seed)
    #     model_noisy_data = RandomForestClassifier(random_state=seed)
    #     model_clean_data_large = RandomForestClassifier(random_state=seed)
    #     model_noisy_data_large = RandomForestClassifier(random_state=seed)
    #
    #     model_cleaned_data.fit(x_train_cleaned, y_train_cleaned)
    #     model_noisy_data.fit(x_train_noisy, y_train_noisy)
    #     model_clean_data.fit(x_train_clean, y_train_clean)
    #     model_noisy_data_large.fit(x_train_large, y_train_large_noisy)
    #     model_clean_data_large.fit(x_train_large, y_train_large_clean)
    #
    #     score_cleaned = model_cleaned_data.score(x_test, y_test)
    #     score_noisy = model_noisy_data.score(x_test, y_test)
    #     score_clean = model_clean_data.score(x_test, y_test)
    #     score_noisy_large = model_noisy_data_large.score(x_test, y_test)
    #     score_clean_large = model_clean_data_large.score(x_test, y_test)
    #
    #     result.append([seed, noise_level,
    #                    score_noisy, score_noisy_large, score_cleaned, score_clean, score_clean_large,
    #                    len(y_train_noisy), len(y_train_large_noisy), len(y_train_cleaned),
    #                    len(y_train_clean), len(y_train_large_clean)])
    #
    # result_df = pd.DataFrame(result, columns=[
    #     "rep", "noise-level", "acc (noisy)", "acc (noisy) (l)", "acc (cleaned)", "acc (clean)", "acc (clean) (l)",
    #     "n (noisy)", "n (noisy) (l)", "n (cleaned)", "n (clean)", "n (clean) (l)"
    # ])
    #
    # melted_df = result_df.melt(id_vars=["rep", "noise-level"],
    #                            value_vars=["acc (noisy)", "acc (noisy) (l)", "acc (cleaned)",
    #                                        "acc (clean)", "acc (clean) (l)"],
    #                            value_name="Accuracy", var_name="Approach")
    #
    # print(melted_df.groupby("Approach").mean())
    #
    # g = sns.catplot(data=melted_df, x="Approach", y="Accuracy", kind="box")
    # for ax in g.axes.flatten():
    #     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    # plt.tight_layout()
    # plt.show()
    #
    #     # print("Cleaned:     {} ({})".format(round(score_cleaned, 2), len(y_train_cleaned)))
    #     # print("Noisy:       {} ({})".format(round(score_noisy, 2), len(y_train_noisy)))
    #     # print("Clean:       {} ({})".format(round(score_clean, 2), len(y_train_clean)))
    #     # print("Noisy (l):   {} ({})".format(round(score_noisy_large, 2), len(y_train_large_noisy)))
    #     # print("Clean (l):   {} ({})".format(round(score_clean_large, 2), len(y_train_large_clean)))
    #     # print()
    #
