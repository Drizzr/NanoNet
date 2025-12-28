import numpy as np
from NanoNet.Exceptions.errors import NetworkConfigError, RegularizationError

class CostFunction:


    def __init__(self, net = None, l1=False, l2=False, lambd=0.0):
        self.l1 = l1
        self.l2 = l2
        self.net = net
        self.lambd = lambd

        if l1 and l2:
            raise RegularizationError("Only one regularization-method can be used at the same time!")
        
    def l2_regularization(self):
        # Standard: 0.5 * lambda * sum of weights squared
        return 0.5 * self.lambd * sum(np.sum(w**2) for w in self.net.weights)

    def l1_regularization(self):
        # Standard: lambda * sum of absolute weights
        return self.lambd * sum(np.sum(np.abs(w)) for w in self.net.weights)