import numpy as np
from NanoNet.Exceptions.errors import NetworkConfigError, RegularizationError

class CostFunction:


    def __init__(self, net = None, l1=False, l2=False, classify=True, lambd=0.0):
        self.classify = classify
        self.l1 = l1
        self.l2 = l2
        self.net = net
        self.lambd = lambd

        if l1 and l2:
            raise RegularizationError("Only one regularization-method can be used at the same time!")
        
        if not self.classify and self.__name__ not in ["QuadraticCost", "MeanAbsoluteCost"]:
            raise NetworkConfigError("Regresssion-type Neural Networks can only be used in combination with the QuadraticCost-Function!")
        

    def l2_regularization(self):
        return self.lambd*np.sum([np.sum(w**2) for w in self.net.weights])
    

    def l1_regularization(self):
        return self.lambd*np.sum([np.sum(np.abs(w)) for w in self.net.weights])