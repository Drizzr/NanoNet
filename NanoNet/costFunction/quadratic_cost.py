import numpy as np
from .base import CostFunction


class QuadraticCost(CostFunction):

    __name__ = "QuadraticCost"

    @staticmethod
    def forward(a, y):

        return 0.5*np.linalg.norm(a-y)**2 

    @staticmethod
    def delta(z, a, y, activation):
        """Return the error delta from the output layer."""
        return (a-y)*activation.derivative(z)