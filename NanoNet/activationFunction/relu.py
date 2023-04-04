import numpy as np


class ReLu:

    def forward(z):
        return np.maximum(0, z)
    
    def derivative(z):
        return np.where(z<=0, 0, 1)
