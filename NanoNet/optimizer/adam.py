import numpy as np
from .base import Optimizer

class ADAM(Optimizer):
    def __init__(self, network: object, cost_function: object, eta=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(network, cost_function)
        
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # t tracks the time step for bias correction
        self.t = 0 

        # Moments for weights (m = first moment/mean, v = second moment/uncentered variance)
        self.m_w = None
        self.v_w = None
        
        # Moments for biases
        self.m_b = None
        self.v_b = None

    def update_mini_batch(self, x, y):
        # 1. Get gradients and current batch loss from the base class
        nabla_b, nabla_w, loss = self.backprop(x, y)

        # 2. Lazy initialization of moment vectors
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in self.NETWORK.weights]
            self.v_w = [np.zeros_like(w) for w in self.NETWORK.weights]
            self.m_b = [np.zeros_like(b) for b in self.NETWORK.biases]
            self.v_b = [np.zeros_like(b) for b in self.NETWORK.biases]

        # 3. Increment time step
        self.t += 1

        # 4. Update parameters
        for i in range(len(self.NETWORK.weights)):
            # --- Update Weight Moments ---
            # m_t = beta1 * m_{t-1} + (1 - beta1) * g
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * nabla_w[i]
            # v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (nabla_w[i]**2)

            # --- Bias Correction ---
            m_hat_w = self.m_w[i] / (1 - self.beta1**self.t)
            v_hat_w = self.v_w[i] / (1 - self.beta2**self.t)

            # Update weights
            self.NETWORK.weights[i] -= self.eta * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)

            # --- Update Bias Moments ---
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * nabla_b[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (nabla_b[i]**2)

            m_hat_b = self.m_b[i] / (1 - self.beta1**self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2**self.t)

            # Update biases
            self.NETWORK.biases[i] -= self.eta * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

        # Return the loss for the TQDM progress bar
        return loss