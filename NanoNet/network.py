import numpy as np
import time 
import json
import sys
import time


class Network:

    def __init__(self, sizes: list, optimizer: object, a_functions: list, cost_function: object, test_data: list = None, training_data:list = None, w_init_size: str ="small"):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y) for y in sizes[1:]]
        self.weights = self.initialize_weights(w_init_size)
        self.test_data = test_data
        self.trainig_data = training_data
        
        self.a_functions = self.initialize_activations(a_functions)

        self.optimizer = self.initialize_optimizer(optimizer, cost_function)

    def initialize_weights(self, size):
        if size.lower() == "small":
            return [0.01*np.random.randn(x, y)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        elif size.lower() == "large":
            return [0.01*np.random.randn(x, y) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def initialize_activations(self, a):
        if len(a) == self.num_layers - 1:
            return a
        raise ValueError
    
    def initialize_optimizer(self, optimizer, cost_function):
        optimizer.WEIGHTS = self.weights
        optimizer.BIASES = self.biases
        optimizer.ACTIVATION_FUNCTIONS = self.a_functions
        optimizer.COST_FUNCTION = cost_function
        optimizer.NUM_LAYERS = self.num_layers
        return optimizer

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    def feedforward(self, a):
        joined = list(zip(self.biases, self.weights))
        
        for i in range(0, self.num_layers-1):
            a = self.a_functions[i].forward(np.dot(joined[i][1].T, a).T+joined[i][0])

        #print(a)
        return a
    
    def evaluate(self):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in self.test_data]
        return sum(int(x == y) for (x, y) in test_results)
    

    def train(self, epochs):
        start_time = time.time()
        if self.test_data: 
            n_test = len(self.test_data)
            print(f"Epoch {0}: {self.evaluate()} / {n_test} ")

        for j in range(epochs):
            self.optimizer.minimize(self.trainig_data)

            self.weights, self.biases = self.optimizer.WEIGHTS, self.optimizer.BIASES


        
            if self.test_data:
                print(f"Epoch {j+1}: {self.evaluate()} / {n_test}")
            else:
                print(f"Epoch {j+1} complete")
        
        print("-----------------------------")
        print(f"finished in {round(time.time() - start_time, 4)} seconds 🥵")       
        print("-----------------------------")

        

def load_from_file(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
        
