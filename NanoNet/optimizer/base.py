import numpy as np

class Optimizer:
    
    WEIGHTS = None
    BIASES  = None
    ACTIVATION_FUNCTIONS = None
    COST_FUNCTION = None
    NUM_LAYERS = None
    MINI_BATCH_SIZE = None

    n = None


    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.BIASES]
        nabla_w = [np.zeros(w.shape) for w in self.WEIGHTS]
        # feedforward
        activation = x
        activations = [x] 
        zs = [] 

        joined = list(zip(self.BIASES,self.WEIGHTS))
        for i in range(0, self.NUM_LAYERS-1):
            z = np.dot(activations[-1], joined[i][1]) + joined[i][0]
            

            zs.append(z)
            activation = self.ACTIVATION_FUNCTIONS[i].forward(z)
            activations.append(activation)

        delta = self.COST_FUNCTION.delta(zs[-1], activations[-1], y, self.ACTIVATION_FUNCTIONS[-1])

        nabla_b[-1] = delta.mean(0)
        nabla_w[-1] = np.dot(delta.T, activations[-2]).T / self.MINI_BATCH_SIZE
        for l in range(2, self.NUM_LAYERS):
            #print(1)
            z = zs[-l]
            sp = self.ACTIVATION_FUNCTIONS[-l].derivative(z)
            delta = np.dot(self.WEIGHTS[-l+1], delta.T).T * sp
            if np.isnan(sp).all():
                print(sp)
                #print(z)
                raise KeyboardInterrupt
            nabla_b[-l] = delta.mean(0)
            #print((self.biases[-l]-nabla_b[-l]).shape)
            nabla_w[-l] = np.dot(delta.T, activations[-l-1]).T / self.MINI_BATCH_SIZE
        return (nabla_b, nabla_w)
    

    def create_minibatch(self, trainig_data):
        controll = []
        mini_batches = []

        for k in range(0, self.n, self.MINI_BATCH_SIZE):
            batch, controll_batch = [], []
            for tupel in trainig_data[k:k+self.MINI_BATCH_SIZE]:
                batch.append(tupel[0])
                controll_batch.append(tupel[1])
                #print(training_data[k:k+mini_batch_size][1])
            mini_batches.append(np.array(batch))
            controll.append(np.array(controll_batch))
        
        return zip(mini_batches, controll)
    

    def minimize(self, trainig_data):

        #random.shuffle(trainig_data)
                
        data = self.create_minibatch(trainig_data)
        for mini_batch, controll in data:
            self.update_mini_batch(mini_batch, controll)

        
    