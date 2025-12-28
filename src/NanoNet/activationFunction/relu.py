import numpy as np

class ReLu:
    
    __name__ = "ReLu"
    
    @staticmethod
    def forward(z):
        """Standard ReLU: max(0, z)"""
        return np.maximum(0, z)
    
    @staticmethod
    def derivative(z):
        """
        Returns the gradient of ReLU.
        Note: We return a NEW array to avoid modifying the 'z' 
        stored during the forward pass.
        """
        # Returns 1.0 where z > 0, and 0.0 otherwise.
        # This is non-destructive.
        return (z > 0).astype(float)