import numpy as np
from .base import Optimizer

class RMSProp(Optimizer):
    """
    RMSProp Optimizer: Maintains a moving average of the squared gradients 
    to normalize the gradient updates.
    """
    def __init__(self, network: object, cost_function: object, eta=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(network, cost_function)

        self.eta = eta
        self.beta = beta
        self.epsilon = epsilon

        # Moving average of squared gradients
        self.v_w = None
        self.v_b = None

    def update_mini_batch(self, x, y):
        """
        Performs the weight/bias update using the RMSProp logic.
        """
        # 1. Unpack the 3 values from the base Optimizer backprop
        nabla_b, nabla_w, loss = self.backprop(x, y)

        # 2. Lazy initialization of state variables
        if self.v_w is None:
            self.v_w = [np.zeros_like(w) for w in self.NETWORK.weights]
            self.v_b = [np.zeros_like(b) for b in self.NETWORK.biases]

        # 3. Update parameters
        for i in range(len(self.NETWORK.weights)):
            # Update moving average of squared gradients for weights
            # v_t = beta * v_{t-1} + (1 - beta) * g^2
            self.v_w[i] = self.beta * self.v_w[i] + (1 - self.beta) * (nabla_w[i]**2)
            
            # Update weights
            self.NETWORK.weights[i] -= self.eta * nabla_w[i] / (np.sqrt(self.v_w[i]) + self.epsilon)

            # Update moving average of squared gradients for biases
            self.v_b[i] = self.beta * self.v_b[i] + (1 - self.beta) * (nabla_b[i]**2)
            
            # Update biases
            self.NETWORK.biases[i] -= self.eta * nabla_b[i] / (np.sqrt(self.v_b[i]) + self.epsilon)

        # 4. Return the loss for the live TQDM progress bar
        return loss