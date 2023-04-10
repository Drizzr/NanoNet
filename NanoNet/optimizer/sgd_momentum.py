import numpy as np
from .base import Optimizer

class SGD_Momentum(Optimizer):
    vel_w = None
    vel_b = None

    def __init__(self, eta, beta=0.9):
        super().__init__()
        self.beta = beta
        self.eta = eta
        

    def minimize(self, trainig_data):

        #random.shuffle(trainig_data)
                
        data = self.create_minibatch(trainig_data)
        for mini_batch, controll in data:
            self.update_mini_batch(mini_batch, controll)
    
    def update_mini_batch(self, mini_batch, controll):
        if not self.vel_w:
            self.vel_b = [np.zeros(b.shape) for b in self.BIASES]
            self.vel_w = [np.zeros(w.shape) for w in self.WEIGHTS]

        delta_nabla_b, delta_nabla_w = self.backprop(mini_batch, controll)
    
        self.vel_w = [self.beta*vel_w+(1-self.beta)*nw for vel_w, nw in zip(self.vel_w, delta_nabla_w)]
        self.vel_b = [self.beta*vel_b+(1-self.beta)*nb for vel_b, nb in zip(self.vel_b, delta_nabla_b)]

        self.WEIGHTS = [w-self.eta*vel_w
                        for w, vel_w in zip(self.WEIGHTS, self.vel_w)]
        
        self.BIASES = [b-self.eta*vel_b
                        for b, vel_b in zip(self.BIASES, self.vel_b)]