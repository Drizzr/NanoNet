import numpy as np

class SoftMax():

    __name__ = "SoftMax"

    @staticmethod
    def forward(z):
        if z.ndim == 1:
            exp_values = np.exp(z- np.max(z))
            probabilities = exp_values / np.sum(exp_values)
            if np.sum(exp_values) == 0:
                raise KeyboardInterrupt
        else:

            exp_values = np.exp(z- np.max(z, axis=1, keepdims=True))
            probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) 
        
        return probabilities
    

