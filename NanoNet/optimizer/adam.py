from .rms_promp import RMSPromp
from .sgd_momentum import SGD_Momentum
import numpy as np


class ADAM(SGD_Momentum, RMSPromp):

    def __init__(self, eta, beta1=0.9, beta2=0.99):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
    

    def update_mini_batch(self, mini_batch, controll):
        if not self.estimated_crude_variance_w:
            self.estimated_crude_variance_b = [np.zeros(b.shape) for b in self.BIASES]
            self.estimated_crude_variance_w = [np.zeros(w.shape) for w in self.WEIGHTS]
            self.vel_b = [np.zeros(b.shape) for b in self.BIASES]
            self.vel_w = [np.zeros(w.shape) for w in self.WEIGHTS]

        delta_nabla_b, delta_nabla_w = self.backprop(mini_batch, controll)
    
        self.estimated_crude_variance_w = [self.beta2*crude+(1-self.beta2)*nw**2 for crude, nw in zip(self.estimated_crude_variance_w, delta_nabla_w)]
        self.estimated_crude_variance_b = [self.beta2*crude+(1-self.beta2)*nb**2 for crude, nb in zip(self.estimated_crude_variance_b, delta_nabla_b)]
        self.vel_w = [self.beta1*vel_w+(1-self.beta1)*nw for vel_w, nw in zip(self.vel_w, delta_nabla_w)]
        self.vel_b = [self.beta1*vel_b+(1-self.beta1)*nb for vel_b, nb in zip(self.vel_b, delta_nabla_b)]


        self.WEIGHTS = [w-self.eta*(1/np.sqrt(crude + self.EPSILON))*vel_w
                        for w, crude, vel_w in zip(self.WEIGHTS, self.estimated_crude_variance_w, self.vel_w)]
        
        self.BIASES = [b-self.eta*(1/np.sqrt(crude + self.EPSILON))*vel_b
                        for b, crude, vel_b in zip(self.BIASES, self.estimated_crude_variance_b, self.vel_b)]