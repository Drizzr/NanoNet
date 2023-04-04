import numpy as np
from NanoNet.activationFunction.sigmoid import Sigmoid



class CostFunction:
    pass

    # l2, l1 regularisation

class CrossEntropy(CostFunction):
    
    def __init__(self, l1 : bool, l2 : bool):
        pass
    @staticmethod
    def forward():
        pass

    @staticmethod
    def delta():
        pass


class LogLikelihood(CostFunction):

    @staticmethod
    def forward():
        pass

    @staticmethod
    def delta():
        pass


class QuadraticCost(CostFunction):

    @staticmethod
    def forward(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * Sigmoid().derivative(z)