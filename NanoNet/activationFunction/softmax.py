import numpy as np

class SoftMax():

    __name__ = "SoftMax"

    @staticmethod
    def forward(z):

        exp_values = np.exp(z- np.max(z, axis=-1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        
        return probabilities
    

