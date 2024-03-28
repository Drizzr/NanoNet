import numpy as np
import time 
import json
import time
from werkzeug.utils import import_string
from NanoNet.Exceptions import LayerConfigError, NetworkConfigError


class Network:

    def __init__(self, sizes: list , a_functions: list,
                w_init_size: str ="small"):
        
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights, self.biases = self.initialize_params(w_init_size)
        self.a_functions = self.initialize_activations(a_functions)

        for i in range(self.num_layers-1):
            if a_functions[i].__name__ == "SoftMax" and i != self.num_layers - 2:
                raise NetworkConfigError("The SoftMax activation-function can only be used in the output-layer!")


    def initialize_params(self, size):
        if size.lower() == "small":
            weights = [np.random.randn(x, y)/np.sqrt(y) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        elif size.lower() == "large":
            weights = [np.random.randn(x, y) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        
        biases = [np.random.randn(y) for y in self.sizes[1:]]
        
        return weights, biases

    def initialize_activations(self, a):
        if len(a) == self.num_layers - 1:
            return a
        raise LayerConfigError("Structure in size attribute and provided activation-functions do not match. (First layer needs no activation-function) \
                            Note that if the network has n-layers you have to define n-1 activation Functions")
    
    def feedforward(self, a):
        joined = list(zip(self.biases, self.weights))
        
        for i in range(0, self.num_layers-1):
            a = self.a_functions[i].forward(np.dot(a, joined[i][1]) + joined[i][0])

        return a
    
    def train(self, epochs : int, training_dataset: object,
            optimizer:object,  step_callback = None, epoch_callback = None):

        if not optimizer.NETWORK:
            optimizer.NETWORK = self


        start_time = time.time()

        print("-----------------------------")
        print(f"Training started 🚀")
        print("model summary:")
        print(f"Layers: {self.num_layers}")
        print(f"Sizes: {self.sizes}")
        print(f"trainable parameters: {sum([w.size for w in self.weights]) + sum([b.size for b in self.biases])}")
        print(f"Activation Functions: {[function.__name__ for function in self.a_functions]}")
        print(f"Optimizer: {optimizer.__class__.__name__}")
        print(f"Cost Function: {optimizer.COST_FUNCTION.__name__}")
        print(f"Learning Rate: {optimizer.eta}")
        
        if optimizer.COST_FUNCTION.l2:
            print(f"Regularization: l2, lambda: {optimizer.COST_FUNCTION.lambd}")
        elif optimizer.COST_FUNCTION.l1:
            print(f"Regularization: l1, lambda: {optimizer.COST_FUNCTION.lambd}")
        else:
            print("Regularization: None")
        print(f"Epochs: {epochs}")
        print("-----------------------------")
        print("default loss: ")
        for j in range(epochs):

            if epoch_callback:
                epoch_callback(epoch=j)

            for index, (x, y) in enumerate(training_dataset):

                optimizer.step(x, y)
                

                if step_callback:
                    step_callback(index=index, epoch=j)
                
            print(f"Epoch {j+1} complete:")

        print("-----------------------------")
        print(f"finished in {round(time.time() - start_time, 4)} seconds 🥵")       
        print("-----------------------------")

    def save(self, filename):
        """
        Save the neural network to the file ``filename``.
        """
        weights = self.best_weights if self.best_weights else self.weights
        biases = self.best_biases if self.best_biases else self.biases

        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in weights],
                "biases": [b.tolist() for b in biases],
                "activation_functions": [str(function.__name__) for function in self.a_functions]}
        
        with open(filename, "w") as f:
            json.dump(data, f)

        print(f"Saved parameters to {filename}!")
    

def load_from_file(filename):
    """
    Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    
    with open(filename, "r") as f:
        data = json.load(f)
        activation_functions = data["activation_functions"]
        activation_functions = [import_string(f"NanoNet.activationFunction.{function}") for function in activation_functions]
    
    net = Network(data["sizes"], activation_functions)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]

    print("Successfully loaded the Network 🚀!")
    return net
        
