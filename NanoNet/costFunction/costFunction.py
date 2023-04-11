import numpy as np
from NanoNet.activationFunction.sigmoid import Sigmoid
import time



class CostFunction:

    def __init__(self, l2=False, l1=False) -> None:
        if l1 and l2:
            raise KeyError
        self.l2 = l2
        self.l1 = l1

    @staticmethod
    def l2_regularisation_forward(weights):
        sum = 0
        for w in weights:
            sum += np.sum(w*w)
        return sum

    @staticmethod
    def l1_regularisation_forward(weights):
        sum = 0
        for w in weights:
            sum += np.sum(np.abs(w))
        return sum



    # l2, l1 regularisation

class CrossEntropy(CostFunction):

    __name__ = "CrossEntropy"

    @staticmethod
    def forward(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        When only a is close to one the function returns inf making learning impossible

        """
        #print(a)
        #time.sleep(5)

        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y, activation=None):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        """print(a)
        print(1)
        print(a-y)
        raise KeyboardInterrupt"""


        return (a-y) # only with sigmoid otherwise division through zero might occure

class LogLikelihood(CostFunction):

    __name__ = "LogLikelihood"

    @staticmethod
    def forward(a, y):

        index = np.argmax(y)
        return -np.log(a[index])

    @staticmethod
    def delta(z, a, y, activation=None):
        return (a-y) # only when last layer uses softmax


class QuadraticCost(CostFunction):

    __name__ = "QuadraticCost"

    @staticmethod
    def forward(a, y):

        return 0.5*np.linalg.norm(a-y)**2 

    @staticmethod
    def delta(z, a, y, activation):
        """Return the error delta from the output layer."""
        return (a-y)*activation.derivative(z)