import numpy as np
from .base import CostFunction

class CategorialCorssEntropy(CostFunction):

    @staticmethod
    def forward(a, y):
        return np.nan_to_num(np.dot(y, -np.log(a)))
    
    @staticmethod
    def delta(z, a, y, activation=None):
        if activation.__name__ == "SoftMax":
            return (a-y)
        else:
            return (-y / a) * activation.derivative(z)
