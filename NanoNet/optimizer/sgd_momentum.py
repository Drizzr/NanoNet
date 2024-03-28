import numpy as np
from .base import Optimizer

class SGD_Momentum(Optimizer):
    vel_w = None
    vel_b = None

    def __init__(self, network : object, cost_function : object, eta, beta=0.9):
        
        super().__init__(network, cost_function)
        self.beta = beta
        self.eta = eta


    
    def update_mini_batch(self, mini_batch, controll):
        if not self.vel_w:
            self.vel_b = [np.zeros(b.shape) for b in self.NETWORK.biases]
            self.vel_w = [np.zeros(w.shape) for w in self.NETWORK.weights]

        delta_nabla_b, delta_nabla_w = self.backprop(mini_batch, controll)

    
        self.vel_w = [self.beta*vel_w+(1-self.beta)*nw for vel_w, nw in zip(self.vel_w, delta_nabla_w)]
        self.vel_b = [self.beta*vel_b+(1-self.beta)*nb for vel_b, nb in zip(self.vel_b, delta_nabla_b)]

        self.NETWORK.weights = [w-self.eta*vel_w
                        for w, vel_w in zip(self.NETWORK.weights, self.vel_w)]
        
        self.NETWORK.biases = [b-self.eta*vel_b
                        for b, vel_b in zip(self.NETWORK.biases, self.vel_b)]
    
