from NanoNet.optimizer.base import Optimizer
import numpy as np

class SGD(Optimizer):

    def __init__(self, learnig_rate, lamb=1):
        self.eta = learnig_rate
        self.lamb = lamb
    
    def update_mini_batch(self, mini_batch, controll):

        delta_nabla_b, delta_nabla_w = self.backprop(mini_batch, controll)

        """if self.COST_FUNCTION.l2:
            weight_decay = 1-self.eta*(self.lamb/self.n)
            self.WEIGHTS = [weight_decay*w-self.eta*nw
                        for w, nw in zip(self.WEIGHTS, delta_nabla_w)]
        
        elif self.COST_FUNCTION.l1:
            self.WEIGHTS = [w-(np.sign(w)*self.eta*self.lamb)/(self.n)-self.eta*nw
                        for w, nw in zip(self.WEIGHTS, delta_nabla_w)]
        
        else:"""
        self.WEIGHTS = [w-self.eta*nw
                            for w, nw in zip(self.WEIGHTS, delta_nabla_w)]

        
        self.BIASES  = [b-self.eta*nb
                        for b, nb in zip(self.BIASES, delta_nabla_b)]
        

