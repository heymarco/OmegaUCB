from abc import ABC, abstractmethod

import numpy as np


class AsymptoticFunction(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class MichaelisMenten(AsymptoticFunction):
    def __call__(self, *args, **kwargs):
        x = args[0]
        f_max = args[1]
        y_50 = args[2]
        power = args[3]
        y = (f_max * np.power(x, power)) / (y_50 + np.power(x, power))
        return y


class M(AsymptoticFunction):
    def __init__(self,
                 m_0: float = 0.1, fit_m_0: bool = True,
                 power: float = 1, fit_power: bool = True):
        self.m_0 = m_0
        self.fit_m_0 = fit_m_0
        self.power = power
        self.fit_power = fit_power

    def __call__(self, x, m_0, f_max, y_50, power):
        m_0 = m_0 if self.fit_m_0 else self.m_0
        power = power if self.fit_power else self.power
        conv = MichaelisMenten()(x, 1, y_50, power)
        y = m_0 + (f_max - m_0) * conv
        return y
