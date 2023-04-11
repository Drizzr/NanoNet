import numpy as np
from .base import CostFunction

class LogLikelihood(CostFunction):

    __name__ = "LogLikelihood"

    @staticmethod
    def forward(a, y):

        index = np.argmax(y)
        return -np.log(a[index])

    @staticmethod
    def delta(z, a, y, activation=None):
        return (a-y) # only when last layer uses softmax