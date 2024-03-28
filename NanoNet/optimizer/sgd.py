from NanoNet.optimizer.base import Optimizer
import numpy as np

class SGD(Optimizer):

    def __init__(self, network : object, cost_function : object, learnig_rate):
        
        super().__init__(network, cost_function)
        self.eta = learnig_rate
    
    def update_mini_batch(self, mini_batch, controll):

        delta_nabla_b, delta_nabla_w = self.backprop(mini_batch, controll)

        self.NETWORK.weights = [w-self.eta*nw
                            for w, nw in zip(self.NETWORK.weights, delta_nabla_w)]

        
        self.NETWORK.biases  = [b-self.eta*nb
                        for b, nb in zip(self.NETWORK.biases, delta_nabla_b)]
        

