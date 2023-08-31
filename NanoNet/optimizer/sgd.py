from NanoNet.optimizer.base import Optimizer
import numpy as np

class SGD(Optimizer):

    def __init__(self, learnig_rate, lambd=1):
        self.eta = learnig_rate
        self.lambd = lambd
    
    def update_mini_batch(self, mini_batch, controll):

        delta_nabla_b, delta_nabla_w = self.backprop(mini_batch, controll)

        self.WEIGHTS = [w-self.eta*nw
                            for w, nw in zip(self.WEIGHTS, delta_nabla_w)]

        
        self.BIASES  = [b-self.eta*nb
                        for b, nb in zip(self.BIASES, delta_nabla_b)]
        

