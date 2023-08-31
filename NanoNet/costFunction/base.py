import numpy as np

class CostFunction:
    @staticmethod
    def l2_regularization(lambd, weights):
        return lambd*np.sum([np.sum(w**2) for w in weights])
    
    @staticmethod
    def l1_regularization(lambd, weights):
        return (lambd)*np.sum([np.sum(np.abs(w)) for w in weights])