import numpy as np
from .base import CostFunction

class LogLikelihood(CostFunction):

    __name__ = "LogLikelihood"

    def __init__(self, net, l1, l2, classify=True, lambd=0.0):
        super().__init__(net, l1, l2, classify, lambd)

    @staticmethod
    def forward(a, y):

        

        return -np.log(a[:,np.argmax(y, axis=-1)])


    @staticmethod
    def delta(z, a, y, activation=None):
        return (a-y) # only when last layer uses softmax