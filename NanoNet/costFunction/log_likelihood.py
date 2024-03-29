import numpy as np
from .base import CostFunction

class LogLikelihood(CostFunction):

    __name__ = "LogLikelihood"

    def __init__(self, net, l1, l2, classify=True, lambd=0.0):
        super().__init__(net, l1, l2, classify, lambd)

    def forward(self, a, y):

        if y.ndim == 1:
            out =  -np.log(a[np.argmax(y)])
        else:
            out = -np.log(a[[i for i in range(len(y))], np.argmax(y, axis=-1)]).mean()

        if self.l1:
            return out + self.l1_regularization()
        
        elif self.l2:
            return out + self.l2_regularization()

        return out


    @staticmethod
    def delta(z, a, y, activation=None):
        return (a-y) # only when last layer uses softmax