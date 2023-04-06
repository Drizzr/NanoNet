import numpy as np
import random
import sys


class Optimizer:
    
    WEIGHTS = None
    BIASES  = None
    ACTIVATION_FUNCTIONS = None
    COST_FUNCTION = None
    NUM_LAYERS = None

    n = 0


class SGD(Optimizer):

    def __init__(self, mini_batch_size, learnig_rate, lamb=1):
        self.mini_batch_size = mini_batch_size
        self.eta = learnig_rate
        self.lamb = lamb

        
    def minimize(self, trainig_data):

        #random.shuffle(trainig_data)

        mini_batches = []
        controll = []

        for k in range(0, self.n, self.mini_batch_size):
            batch, controll_batch = [], []
            for tupel in trainig_data[k:k+self.mini_batch_size]:
                batch.append(tupel[0])
                controll_batch.append(tupel[1])
                #print(training_data[k:k+mini_batch_size][1])
            mini_batches.append(np.array(batch))
            controll.append(np.array(controll_batch))
                
        data = zip(mini_batches, controll)
        for mini_batch, controll in data:
            #print(mini_batch)
            #print(np.array(controll).shape)
            self.update_mini_batch(mini_batch, controll)

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.BIASES]
        nabla_w = [np.zeros(w.shape) for w in self.WEIGHTS]
        # feedforward
        activation = x
        activations = [x] 
        zs = [] 

        joined = list(zip(self.BIASES,self.WEIGHTS))
        for i in range(0, self.NUM_LAYERS-1):
            z = np.dot(activations[-1], joined[i][1]) + joined[i][0]
            

            zs.append(z)
            activation = self.ACTIVATION_FUNCTIONS[i].forward(z)
            activations.append(activation)

        delta = self.COST_FUNCTION.delta(zs[-1], activations[-1], y, self.ACTIVATION_FUNCTIONS[-1])

        nabla_b[-1] = delta.mean(0)
        nabla_w[-1] = np.dot(delta.T, activations[-2]).T / self.mini_batch_size
        for l in range(2, self.NUM_LAYERS):
            #print(1)
            z = zs[-l]
            sp = self.ACTIVATION_FUNCTIONS[-l].derivative(z)
            delta = np.dot(self.WEIGHTS[-l+1], delta.T).T * sp
            if np.isnan(sp).all():
                print(sp)
                #print(z)
                raise KeyboardInterrupt
            nabla_b[-l] = delta.mean(0)
            #print((self.biases[-l]-nabla_b[-l]).shape)
            nabla_w[-l] = np.dot(delta.T, activations[-l-1]).T / self.mini_batch_size
        return (nabla_b, nabla_w)

    def update_mini_batch(self, mini_batch, controll):


        delta_nabla_b, delta_nabla_w = self.backprop(mini_batch, controll)

        if self.COST_FUNCTION.l2:
            weight_decay = 1-self.eta*(self.lamb/self.n)
            #print(weight_decay)
            self.WEIGHTS = [weight_decay*w-self.eta*nw
                        for w, nw in zip(self.WEIGHTS, delta_nabla_w)]
        
        elif self.COST_FUNCTION.l1:
            self.WEIGHTS = [w-(np.sign(w)*self.eta*self.lamb)/(self.n)-self.eta*nw
                        for w, nw in zip(self.WEIGHTS, delta_nabla_w)]
        
        else:
            self.WEIGHTS = [w-self.eta*nw
                        for w, nw in zip(self.WEIGHTS, delta_nabla_w)]

        
        self.BIASES  = [b-self.eta*nb
                        for b, nb in zip(self.BIASES, delta_nabla_b)]
        
        #print(delta_nabla_b)
        #print(self.WEIGHTS)
