import numpy as np
import random
from NanoNet.Exceptions import RegularizationError, NetworkConfigError

class Optimizer:
    
    def __init__(self, network : object, cost_function : object) -> None:
        
        self.NETWORK = network
        self.COST_FUNCTION = cost_function

        
        if cost_function.__name__ in ["CategorialCrossEntropy", "LogLikelihood"] and self.NETWORK.a_functions[-1].__name__ != "SoftMax":
            raise NetworkConfigError("The LogLikelihood, CategorialCrossEntropy cost-function can only be used in combination with a sofMax-ouput layer!")

        if cost_function.__name__ == "CrossEntropy" and self.NETWORK.a_functions[-1].__name__ != "Sigmoid":
            raise NetworkConfigError("The BinaryCrossEntropy cost-function can only be used in combination with a sigmoid-ouput layer!")
        
        if self.NETWORK.a_functions[-1].__name__ == "SoftMax" and cost_function.__name__ not in ["CategorialCrossEntropy", "LogLikelihood"]:
            raise NetworkConfigError("The SoftMax activation-function can only be used in combination with the Loglikelihood or CategorialCrossEntropy-cost-function!")
        

    def backprop(self, x, y):

        mini_batch_size = x.shape[0]
        
        nabla_b = [np.zeros(b.shape) for b in self.NETWORK.biases]
        nabla_w = [np.zeros(w.shape) for w in self.NETWORK.weights]
        # feedforward
        activation = x
        activations = [x] 
        zs = [] 

        joined = list(zip(self.NETWORK.biases,self.NETWORK.weights))
        for i in range(0, self.NETWORK.num_layers-1):
            z = np.dot(activations[-1], joined[i][1]) + joined[i][0]
            zs.append(z)

            activation = self.NETWORK.a_functions[i].forward(z)
            activations.append(activation)

        
        delta = self.COST_FUNCTION.delta(zs[-1], activations[-1], y, self.NETWORK.a_functions[-1])


        nabla_b[-1] = delta.mean(0)
        nabla_w[-1] = np.dot(delta.T, activations[-2]).T / mini_batch_size

        for l in range(2, self.NETWORK.num_layers):
            z = zs[-l]
            sp = self.NETWORK.a_functions[-l].derivative(z)
            delta = np.dot(self.NETWORK.weights[-l+1], delta.T).T * sp

            nabla_b[-l] = delta.mean(0)
            #print((self.NETWORK.biases[-l]-nabla_b[-l]).shape)
            nabla_w[-l] = np.dot(delta.T, activations[-l-1]).T / mini_batch_size

        if self.COST_FUNCTION.l2:
            for i in range(len(nabla_w)):
                nabla_w[i] += self.COST_FUNCTION.lambd/mini_batch_size * self.NETWORK.weights[i]
        elif self.COST_FUNCTION.l1:
            for i in range(len(nabla_w)):
                nabla_w[i] += self.COST_FUNCTION.lambd/mini_batch_size * np.sign(self.NETWORK.weights[i])

        return (nabla_b, nabla_w)


    def step(self, x, y):
        self.update_mini_batch(x, y)

        
    