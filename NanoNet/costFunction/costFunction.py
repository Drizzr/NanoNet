import numpy as np
from NanoNet.activationFunction.sigmoid import Sigmoid



class CostFunction:
    pass

    # l2, l1 regularisation

class CrossEntropy(CostFunction):

    @staticmethod
    def forward(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y, activation=None):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y) # only with sigmoid otherwise division through zero might occure

class LogLikelihood(CostFunction):

    @staticmethod
    def forward(a, y):
        index = y.argmax(y)
        return np.log(a[index])

    @staticmethod
    def delta(z, a, y, activation=None):
        return (a-y) # only when last layer uses softmax


class QuadraticCost(CostFunction):

    @staticmethod
    def forward(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y, activation):
        """Return the error delta from the output layer."""
        return (a-y)*activation.derivative(z)