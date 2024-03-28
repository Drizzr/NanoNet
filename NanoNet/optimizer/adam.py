from NanoNet.optimizer.base import Optimizer
import numpy as np


class ADAM(Optimizer):

    estimated_crude_variance_b = None
    estimated_crude_variance_w = None
    estimated_crude_variance_w = None
    estimated_crude_variance_b = None
    EPSILON = 10e-7
    vel_w = None
    vel_b = None

    def __init__(self, network : object, cost_function : object, eta, beta1=0.9, beta2=0.99):

        super().__init__(network, cost_function)

        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
    

    def update_mini_batch(self, mini_batch, controll):
        if not self.estimated_crude_variance_w:
            self.estimated_crude_variance_b = [np.zeros(b.shape) for b in self.NETWORK.biases]
            self.estimated_crude_variance_w = [np.zeros(w.shape) for w in self.NETWORK.weights]
            self.vel_b = [np.zeros(b.shape) for b in self.NETWORK.biases]
            self.vel_w = [np.zeros(w.shape) for w in self.NETWORK.weights]

        delta_nabla_b, delta_nabla_w = self.backprop(mini_batch, controll)
    
        self.estimated_crude_variance_w = [self.beta2*crude+(1-self.beta2)*nw**2 for crude, nw in zip(self.estimated_crude_variance_w, delta_nabla_w)]
        self.estimated_crude_variance_b = [self.beta2*crude+(1-self.beta2)*nb**2 for crude, nb in zip(self.estimated_crude_variance_b, delta_nabla_b)]
        self.vel_w = [self.beta1*vel_w+(1-self.beta1)*nw for vel_w, nw in zip(self.vel_w, delta_nabla_w)]
        self.vel_b = [self.beta1*vel_b+(1-self.beta1)*nb for vel_b, nb in zip(self.vel_b, delta_nabla_b)]


        self.NETWORK.weights = [w-self.eta*(1/np.sqrt(crude + self.EPSILON))*vel_w
                        for w, crude, vel_w in zip(self.NETWORK.weights, self.estimated_crude_variance_w, self.vel_w)]
        
        self.NETWORK.biases = [b-self.eta*(1/np.sqrt(crude + self.EPSILON))*vel_b
                        for b, crude, vel_b in zip(self.NETWORK.biases, self.estimated_crude_variance_b, self.vel_b)]

            

