import numpy as np


class Aggregate:
    """Follows Welford's algorithm to efficiently track the second moment"""

    def __init__(self):
        self._count = 0
        self._mean = 0.0
        self._m2 = 0

    def update(self, newval: float):
        self._count += 1
        delta = newval - self._mean
        self._mean += delta / self._count
        delta2 = newval - self._mean
        self._m2 += delta * delta2

    def variance(self):
        if self._count < 2:
            return 1 / 4
        return self._m2 / (self._count - 1)

    def mean(self):
        return self._mean

    def std(self):
        return np.sqrt(self.variance())
