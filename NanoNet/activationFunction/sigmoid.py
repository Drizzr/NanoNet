import numpy as np

class Sigmoid:

    __name__ = "Sigmoid"

    @staticmethod
    def forward(z):
        #if not np.all(1.0/(1.0+np.exp(-z))):
            #print(1.0/(1.0+np.exp(-z)))

        #print((1.0/(1.0+np.exp(-z))).shape)
        #print(z)
        #print(1.0/(1.0+np.exp(-z)))

        return 1.0/(1.0+np.exp(-z))
    
    @staticmethod
    def derivative(z):

        return 1.0/(1.0+np.exp(-z))*(1-1.0/(1.0+np.exp(-z)))