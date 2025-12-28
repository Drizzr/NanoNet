import numpy as np
from .base import CostFunction

class CategorialCrossEntropy(CostFunction):

    __name__ = "CategorialCrossEntropy"

    def __init__(self, net, l1, l2, lambd=0.0):
        super().__init__(net, l1, l2, lambd)

    def forward(self, a, y):
        """
        Return the cost associated with an output ``a`` and desired output ``y``.
        
        Uses element-wise multiplication for speed and np.clip for stability.
        """
        # 1. Numerical stability: prevent log(0)
        epsilon = 1e-15
        a_clipped = np.clip(a, epsilon, 1.0 - epsilon)

        # 2. Calculate Cross Entropy
        # Formula: -sum( y * log(a) )
        # Using * (element-wise) is much faster than np.dot + np.diagonal
        if a.ndim == 1:
            # Single sample case
            out = -np.sum(y * np.log(a_clipped))
        else:
            # Batch case: Sum across neurons (axis -1), then average the batch
            out = -np.sum(y * np.log(a_clipped), axis=-1).mean()

        # 3. Add Regularization
        if self.l1:
            out += self.l1_regularization()
        elif self.l2:
            out += self.l2_regularization()
            
        return out
    
    @staticmethod
    def delta(z, a, y, activation=None):
        """
        Derivative of Categorical Cross Entropy with respect to the input (z)
        of a Softmax output layer.
        """
        return (a - y) / a.shape[0]