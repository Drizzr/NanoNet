from .base import CostFunction
import numpy as np


class BinaryCrossEntropy(CostFunction):

    __name__ = "CrossEntropy"

    def __init__(self, net, l1, l2, classify=True, lambd=0.0):
        super().__init__(net, l1, l2, classify, lambd)

    def forward(self, a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        When only a is close to one the function returns inf making learning impossible

        """
        if self.l1:
            return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)), axis=-1).mean() + self.l1_regularization()
        elif self.l2:
            return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)), axis=-1).mean() + self.l2_regularization()

        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)), axis=-1).mean()

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