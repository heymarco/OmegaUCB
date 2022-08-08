import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

MNIST = 554
MUSHROOM = 24


if __name__ == '__main__':
    split = 0.01
    clean_data, clean_labels = fetch_openml(data_id=MUSHROOM, return_X_y=True, as_frame=False)
    not_na_rows = np.any(np.isnan(clean_data), axis=1) == False
    clean_data = clean_data[not_na_rows]
    clean_labels = clean_labels[not_na_rows]
    shuffled_indices = np.arange(len(clean_labels))
    np.random.shuffle(shuffled_indices)
    clean_data = clean_data[shuffled_indices]
    clean_labels = clean_labels[shuffled_indices]
    clean_labels = LabelEncoder().fit_transform(clean_labels)
    n_train = int(len(clean_data) * split)
    x_train, x_test = clean_data[:n_train], clean_data[n_train:]
    y_train, y_test = clean_labels[:n_train], clean_labels[n_train:]

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    theta_mle = []
    acc_scores = []
    empirical_error = []
    sample_range = range(10, 10000, 10)
    for sample_size in sample_range:
        sample_indices = np.random.choice(np.arange(len(x_test)), sample_size)
        predictions = model.predict(x_test[sample_indices])
        labels = y_test[sample_indices]
        acc_scores.append(model.score(x_test[sample_indices], labels))
        theta = np.average(predictions)
        theta_mle.append(theta)
        empirical_error.append(
            np.abs(theta - np.average(labels))
        )
    theta_mle = np.array(theta_mle)
    acc_scores = np.array(acc_scores)

    p_y = np.average(clean_labels)
    error = np.abs(theta_mle - p_y)
    true_acc = model.score(x_test, y_test)

    plt.plot(sample_range, error, color="blue", label="predicted error")
    plt.plot(sample_range, empirical_error, color="orange", label="empirical error")
    plt.plot(sample_range, np.abs(true_acc - acc_scores), color="red", label="true error")
    plt.xlabel("sample_size")
    plt.ylabel("risk")
    plt.legend()
    plt.tight_layout(pad=.5)
    plt.show()
