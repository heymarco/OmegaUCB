from .abstract import Oracle
import numpy as np


class SymmetricNoiseOracle(Oracle):

    def __init__(self,
                 true_labels: np.ndarray,
                 noise_level: float,
                 seed: int):
        self.rng = np.random.default_rng(seed)
        self.true_labels = true_labels
        self.num_labels = len(np.unique(true_labels))
        self.noise_level = noise_level

    def get_noisy_label(self, instance: np.ndarray, index: int) -> int:
        true_label = self.true_labels[index]
        random_number = self.rng.uniform(0, 1)
        if random_number < self.noise_level:
            random_label = self.rng.integers(0, self.num_labels)
            while random_label == true_label:
                random_label = self.rng.integers(0, self.num_labels)
            return random_label
        else:
            return true_label

    def get_clean_label(self, index: int) -> int:
        return self.true_labels[index]
