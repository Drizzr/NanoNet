import numpy as np


class ReLu:
    
    @staticmethod
    def forward(z):
        return np.maximum(0, z)
    
    @staticmethod
    def derivative(z):
        z[z<=0] = 0
        z[z>0] = 1
        return z
