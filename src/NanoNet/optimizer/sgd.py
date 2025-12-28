import numpy as np
from .base import Optimizer

class SGD(Optimizer):
    """
    Standard Stochastic Gradient Descent.
    Updates parameters by moving in the opposite direction of the gradient.
    """
    def __init__(self, network: object, cost_function: object, learning_rate: float):
        """
        Args:
            network: The neural network instance.
            cost_function: The cost function instance.
            learning_rate: The step size (eta).
        """
        super().__init__(network, cost_function)
        self.eta = learning_rate
    
    def update_mini_batch(self, x, y):
        """
        Performs a single gradient descent step on a mini-batch.
        """
        # 1. Unpack gradients and the batch loss from backprop
        nabla_b, nabla_w, loss = self.backprop(x, y)

        # 2. Update weights and biases
        for i in range(len(self.NETWORK.weights)):
            self.NETWORK.weights[i] -= self.eta * nabla_w[i]
            self.NETWORK.biases[i] -= self.eta * nabla_b[i]

        # 3. Return the loss for the TQDM progress bar in Network.train
        return loss