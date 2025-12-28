import numpy as np
from .base import Optimizer

class SGD_Momentum(Optimizer):
    """
    Stochastic Gradient Descent with Momentum.
    Accelerates SGD by navigating along relevant directions and softening oscillations.
    """
    def __init__(self, network: object, cost_function: object, eta: float, beta: float = 0.9):
        """
        Args:
            network: The neural network instance.
            cost_function: The cost function instance.
            eta: The learning rate.
            beta: The momentum coefficient (usually 0.9).
        """
        super().__init__(network, cost_function)
        self.eta = eta
        self.beta = beta

        # Velocity state (initialized lazily)
        self.v_w = None
        self.v_b = None

    def update_mini_batch(self, x, y):
        """
        Updates weights and biases using momentum.
        """
        # 1. Unpack gradients and batch loss
        nabla_b, nabla_w, loss = self.backprop(x, y)

        # 2. Lazy initialization of velocity vectors
        if self.v_w is None:
            self.v_w = [np.zeros_like(w) for w in self.NETWORK.weights]
            self.v_b = [np.zeros_like(b) for b in self.NETWORK.biases]

        # 3. Update parameters using the momentum formula
        for i in range(len(self.NETWORK.weights)):
            # Update Weight Velocity: v = beta * v + (1 - beta) * gradient
            # (Note: Using the 1-beta scaling is standard for moving averages)
            self.v_w[i] = self.beta * self.v_w[i] + (1 - self.beta) * nabla_w[i]
            self.NETWORK.weights[i] -= self.eta * self.v_w[i]

            # Update Bias Velocity
            self.v_b[i] = self.beta * self.v_b[i] + (1 - self.beta) * nabla_b[i]
            self.NETWORK.biases[i] -= self.eta * self.v_b[i]

        # 4. Return loss for the TQDM progress bar
        return loss