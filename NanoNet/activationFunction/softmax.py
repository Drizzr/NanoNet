import numpy as np

class SoftMax():

    @staticmethod
    def forward(z):

        exp_values = np.exp(z)

        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        return probabilities
    

