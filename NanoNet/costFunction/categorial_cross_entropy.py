import numpy as np
from .base import CostFunction

class CategorialCorssEntropy(CostFunction):

    __name__ = "CategorialCorssEntropy"


    @staticmethod
    def forward(a, y):
        return np.nan_to_num(np.dot(y, -np.log(a)))
    
    @staticmethod
    def delta(z, a, y, activation=None):
        return (a-y)
