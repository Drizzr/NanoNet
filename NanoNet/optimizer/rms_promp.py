from .base import Optimizer
import numpy as np

class RMSPromp(Optimizer):
    
    estimated_crude_variance_w = None
    estimated_crude_variance_b = None
    EPSILON = 10e-7

    def __init__(self, network : object, cost_function : object, eta, beta=0.99):

        super().__init__(network, cost_function)

        self.eta = eta
        self.beta = beta

    def update_mini_batch(self, mini_batch, controll):
        if not self.estimated_crude_variance_w:
            self.estimated_crude_variance_b = [np.zeros(b.shape) for b in self.NETWORK.biases]
            self.estimated_crude_variance_w = [np.zeros(w.shape) for w in self.NETWORK.weights]

        delta_nabla_b, delta_nabla_w = self.backprop(mini_batch, controll)
    
        self.estimated_crude_variance_w = [self.beta*crude+(1-self.beta)*nw**2 for crude, nw in zip(self.estimated_crude_variance_w, delta_nabla_w)]
        self.estimated_crude_variance_b = [self.beta*crude+(1-self.beta)*nb**2 for crude, nb in zip(self.estimated_crude_variance_b, delta_nabla_b)]

        self.NETWORK.weights = [w-self.eta*(1/np.sqrt(crude + self.EPSILON))*nw
                        for w, crude, nw in zip(self.NETWORK.weights, self.estimated_crude_variance_w, delta_nabla_w)]
        
        self.NETWORK.biases = [b-self.eta*(1/np.sqrt(crude + self.EPSILON))*nb
                        for b, crude, nb in zip(self.NETWORK.biases, self.estimated_crude_variance_b, delta_nabla_b)]

        