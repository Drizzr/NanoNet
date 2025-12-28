import numpy as np

class SoftMax:

    __name__ = "SoftMax"

    @staticmethod
    def forward(z):
        """
        Compute the Softmax activation.
        Uses the max-subtraction trick to prevent overflow (exp(large_number)).
        """
        # Subtracting the max value from each row makes the largest value 0.
        # This ensures all exp values are between 0 and 1.
        e_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return e_z / np.sum(e_z, axis=-1, keepdims=True)
    
    @staticmethod
    def derivative(z):
        """
        Compute the element-wise derivative of Softmax.
        Note: Technically, Softmax has a Jacobian matrix derivative.
        This element-wise version is provided to keep the Optimizer.backprop 
        loop from crashing and works for basic implementations.
        """
        s = SoftMax.forward(z)
        return s * (1 - s)