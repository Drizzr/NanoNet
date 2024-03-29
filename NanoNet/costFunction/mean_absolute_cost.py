from .base import CostFunction
import numpy as np


class MeanAbsoluteCost(CostFunction):

    __name__ = "MeanAbsoluteCost"

    def __init__(self, net, l1, l2, classify=True, lambd=0.0):
        super().__init__(net, l1, l2, classify, lambd)


    def forward(self, a, y):

        if self.l1:
            return np.nan_to_num(np.absolute(y-a).mean(axis=-1)) + self.l1_regularization()
        elif self.l2:
            return np.nan_to_num(np.absolute(y-a).mean(axis=-1)) + self.l2_regularization()

        return np.nan_to_num(np.absolute(y-a).mean())

    @staticmethod
    def delta(z, a, y, activation):
        """Return the error delta from the output layer."""
        return np.sign(y-a)*activation.derivative(z)