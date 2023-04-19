from .base import CostFunction
import numpy as np


class MeanAbsoluteCost(CostFunction):

    __name__ = "MeanAbsoluteCost"


    @staticmethod
    def forward(a, y):

        return np.absolute(y-a)

    @staticmethod
    def delta(z, a, y, activation):
        """Return the error delta from the output layer."""
        return np.sign(y-a)*activation.derivative(z)