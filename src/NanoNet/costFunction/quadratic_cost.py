import numpy as np
from .base import CostFunction


class QuadraticCost(CostFunction):

    __name__ = "QuadraticCost"

    def __init__(self, net, l1, l2, lambd=0.0):
        super().__init__(net, l1, l2, lambd)

    def forward(self, a, y):
        if self.l1:
            return (0.5*np.linalg.norm(a-y, axis=-1)**2).mean() + self.l1_regularization()
        elif self.l2:
            return (0.5*np.linalg.norm(a-y, axis=-1)**2).mean() + self.l2_regularization()

        return (0.5*np.linalg.norm(a-y, axis=-1)**2).mean()

    
    def delta(self, z, a, y, activation):
        """Return the error delta from the output layer."""
        return (a-y)*activation.derivative(z)