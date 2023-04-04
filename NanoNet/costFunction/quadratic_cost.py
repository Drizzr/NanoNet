import numpy as np
from NanoNet.activationFunction.sigmoid import Sigmoid

class QuadraticCost:

    @staticmethod
    def forward(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * Sigmoid().derivative(z)