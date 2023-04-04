import numpy as np

class Sigmoid:
    
    def forward(self, z):
        return 1.0/(1.0+np.exp(-z))
    
    def derivative(self, z):
        return self.forward(z)*(1-self.forward(z))