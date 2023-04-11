import numpy as np

class CostFunction:

    @staticmethod
    def l2_regularisation_forward(weights):
        sum = 0
        for w in weights:
            sum += np.sum(w*w)
        return sum

    @staticmethod
    def l1_regularisation_forward(weights):
        sum = 0
        for w in weights:
            sum += np.sum(np.abs(w))
        return sum