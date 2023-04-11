import numpy as np
import time 
import json
import sys
import time
from werkzeug.utils import import_string


class Network:

    def __init__(self, sizes: list , a_functions: list, cost_function: object, optimizer: object = None, mini_batch_size : int = None, test_data: list = None, training_data:list = None, w_init_size: str ="small"):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y) for y in sizes[1:]]
        self.weights = self.initialize_weights(w_init_size)
        self.test_data = test_data
        self.training_data = training_data
        self.n_test = len(test_data)
        self.n_trainig = len(training_data)
        self.a_functions = self.initialize_activations(a_functions)
        self.cost_function = cost_function

        if optimizer and cost_function:
            self.optimizer = self.initialize_optimizer(optimizer, mini_batch_size)


    def initialize_weights(self, size):
        if size.lower() == "small":
            return [np.random.randn(x, y)/np.sqrt(y) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        elif size.lower() == "large":
            return [np.random.randn(x, y) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def initialize_activations(self, a):
        if len(a) == self.num_layers - 1:
            return a
        raise ValueError
    
    def initialize_optimizer(self, optimizer, mini_batch_size):
        optimizer.WEIGHTS = self.weights
        optimizer.BIASES = self.biases
        optimizer.ACTIVATION_FUNCTIONS = self.a_functions
        optimizer.COST_FUNCTION = self.cost_function
        optimizer.NUM_LAYERS = self.num_layers
        optimizer.n = len(self.training_data)
        optimizer.MINI_BATCH_SIZE = mini_batch_size
        return optimizer


    def feedforward(self, a):
        joined = list(zip(self.biases, self.weights))
        
        for i in range(0, self.num_layers-1):
            a = self.a_functions[i].forward(np.dot(joined[i][1].T, a).T+joined[i][0])

        #print(a)
        return a
    
    def evaluate(self, data, convert=True):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        if convert:
            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
    

    def train(self, epochs, monitor_training_cost=True, monitor_training_accuracy=True, monitor_test_cost=True, monitor_test_accuracy=True):
        start_time = time.time()

        print(f"Epoch {0}: {self.evaluate(self.test_data)} / {self.n_test} ")

        test_cost, test_accuracy = [], []
        training_cost, training_accuracy = [], []

        for j in range(epochs):
            self.optimizer.minimize(self.training_data)

            self.weights, self.biases = self.optimizer.WEIGHTS, self.optimizer.BIASES


            print(f"Epoch {j+1} complete:")
            if monitor_training_cost:
                cost = self.total_cost(self.training_data, convert=False)
                training_cost.append(cost)
                print(f"Epoch {j+1}: Cost on training data: {cost}")
            if monitor_training_accuracy:
                accuracy = self.evaluate(self.training_data, convert=False)
                training_accuracy.append(accuracy)
                print(f"Epoch {j+1}: {accuracy} / {self.n_trainig} ({round((accuracy/self.n_trainig)*100, 2)}%)")
            if monitor_test_cost:
                cost = self.total_cost(self.test_data)
                test_cost.append(cost)
                print(f"Epoch {j+1}: Cost on test data: {cost}")
            if monitor_test_accuracy:
                accuracy = self.evaluate(self.test_data)
                test_accuracy.append(accuracy)
                print(f"Epoch {j+1}: {accuracy} / {self.n_test} ({round((accuracy/self.n_test)*100, 2)}%)")
        
        print("-----------------------------")
        print(f"finished in {round(time.time() - start_time, 4)} seconds 🥵")       
        print("-----------------------------")


    def total_cost(self, data, convert=True):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y, 10)
            cost += self.cost_function.forward(a, y)/ len(data)
        
        if self.cost_function.l2:
            cost += (self.cost_function.l2_regularisation_forward(self.weights)*self.optimizer.lamb/2)/ len(data)
        elif self.cost_function.l1:
            cost += (self.cost_function.l1_regularisation_forward(self.weights)*self.optimizer.lamb)/ len(data)
        
        return round(cost,4)
    
    
    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "activation_functions": [str(function.__name__) for function in self.a_functions],
                "cost_function": str(self.optimizer.COST_FUNCTION.__name__)}
        
        with open(filename, "w") as f:
            json.dump(data, f)
        

def load_from_file(filename, trainig_data, test_data):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    with open(filename, "r") as f:
        data = json.load(f)
        cost_function = data["cost_function"]
        activation_functions = data["activation_functions"]
        activation_functions = [import_string(f"NanoNet.activationFunction.{function}") for function in activation_functions]
        cost_function = import_string(f"NanoNet.costFunction.{cost_function}")
    net = Network(data["sizes"], activation_functions, cost_function, training_data=trainig_data, test_data=test_data)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
        

def vectorized_result(j, length):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros(length)
    e[j] = 1.0
    return e