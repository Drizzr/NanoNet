from .base import Optimizer
import numpy as np

class RMSPromp(Optimizer):
    
    estimated_crude_variance_w = None
    estimated_crude_variance_b = None
    EPSILON = 10e-7

    def __init__(self, eta, beta=0.99):
        self.eta = eta
        self.beta = beta

    def minimize(self, trainig_data):

        #random.shuffle(trainig_data)
                
        data = self.create_minibatch(trainig_data)
        for mini_batch, controll in data:
            self.update_mini_batch(mini_batch, controll)
    

    def update_mini_batch(self, mini_batch, controll):
        if not self.estimated_crude_variance_w:
            self.estimated_crude_variance_b = [np.zeros(b.shape) for b in self.BIASES]
            self.estimated_crude_variance_w = [np.zeros(w.shape) for w in self.WEIGHTS]

        delta_nabla_b, delta_nabla_w = self.backprop(mini_batch, controll)
    
        self.estimated_crude_variance_w = [self.beta*crude+(1-self.beta)*nw**2 for crude, nw in zip(self.estimated_crude_variance_w, delta_nabla_w)]
        self.estimated_crude_variance_b = [self.beta*crude+(1-self.beta)*nb**2 for crude, nb in zip(self.estimated_crude_variance_b, delta_nabla_b)]

        self.WEIGHTS = [w-self.eta*(1/np.sqrt(crude + self.EPSILON))*nw
                        for w, crude, nw in zip(self.WEIGHTS, self.estimated_crude_variance_w, delta_nabla_w)]
        
        self.BIASES = [b-self.eta*(1/np.sqrt(crude + self.EPSILON))*nb
                        for b, crude, nb in zip(self.BIASES, self.estimated_crude_variance_b, delta_nabla_b)]
        