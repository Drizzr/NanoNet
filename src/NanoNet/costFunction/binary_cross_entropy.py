from .base import CostFunction
import numpy as np


class BinaryCrossEntropy(CostFunction):

    __name__ = "CrossEntropy"

    def __init__(self, net, l1, l2, lambd=0.0):
        super().__init__(net, l1, l2, lambd)

    def forward(self, a, y):
        """
        Return the cost associated with an output ``a`` and desired output ``y``.
        
        Uses np.clip to ensure numerical stability by preventing log(0) which
        results in infinity.
        """
        # 1. Numerical stability: clip 'a' so it's never exactly 0 or 1
        # 1e-15 is a standard epsilon value for this purpose
        epsilon = 1e-15
        a_clipped = np.clip(a, epsilon, 1.0 - epsilon)

        # 2. Calculate the core Binary Cross Entropy cost
        # We calculate the sum of errors for each sample, then take the mean of the batch
        # Formula: -[y*log(a) + (1-y)*log(1-a)]
        sample_costs = - (y * np.log(a_clipped) + (1 - y) * np.log(1 - a_clipped))
        
        # Sum over the last axis (output neurons) and mean over the first (batch size)
        cost = np.sum(sample_costs, axis=-1).mean()

        # 3. Add Regularization (if any)
        if self.l1:
            cost += self.l1_regularization()
        elif self.l2:
            cost += self.l2_regularization()

        return cost

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