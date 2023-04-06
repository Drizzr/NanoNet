import numpy as np

class Sigmoid:
    
    def forward(self, z):
        #if not np.all(1.0/(1.0+np.exp(-z))):
            #print(1.0/(1.0+np.exp(-z)))

        #print((1.0/(1.0+np.exp(-z))).shape)
        #print(z)
        #print(1.0/(1.0+np.exp(-z)))

        return 1.0/(1.0+np.exp(-z))
    
    def derivative(self, z):

        return self.forward(z)*(1-self.forward(z))