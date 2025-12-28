import numpy as np

class Sigmoid:

    __name__ = "Sigmoid"

    @staticmethod
    def forward(z):
        """
        Compute the sigmoid function.
        Uses np.clip to prevent overflow errors with large negative numbers.
        """
        # We clip z to stay within a range where np.exp won't overflow.
        # -500 to 500 is plenty for floating point precision.
        z_safe = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_safe))
    
    @staticmethod
    def derivative(z):
        """
        Compute the derivative of the sigmoid function.
        f'(z) = f(z) * (1 - f(z))
        """
        # Clip here as well for stability
        z_safe = np.clip(z, -500, 500)
        s = 1.0 / (1.0 + np.exp(-z_safe))
        return s * (1 - s)