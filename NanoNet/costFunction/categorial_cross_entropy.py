import numpy as np
from .base import CostFunction

class CategorialCrossEntropy(CostFunction):

    __name__ = "CategorialCrossEntropy"

    def __init__(self, net, l1, l2, classify=True, lambd=0.0):
        super().__init__(net, l1, l2, classify, lambd)


    def forward(self, a, y):
        if a.ndim != 1:
            out =  np.nan_to_num(np.diagonal(np.dot(y, -np.log(a.T)))).mean()
        else:
            out = np.nan_to_num(np.dot(y, -np.log(a))).mean()
        if self.l1:
            return out + self.l1_regularization()
        elif self.l2:
            return out + self.l2_regularization()
        return out
    
    @staticmethod
    def delta(z, a, y, activation=None):
        return (a-y)/a.shape[0]
