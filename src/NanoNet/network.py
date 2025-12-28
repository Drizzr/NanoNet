import numpy as np
import time 
import json
import os
from typing import List, Optional, Callable
# We removed werkzeug and now import our activations directly for loading
import NanoNet.activationFunction as af
from NanoNet.Exceptions import LayerConfigError, NetworkConfigError
from tqdm import tqdm

class Network:
    def __init__(self, sizes: List[int], a_functions: List[object], w_init: str = "xavier", dropout_rate: float = 0.0):
        """
        Initializes the Neural Network.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.a_functions = self.initialize_activations(a_functions)
        self.weights, self.biases = self.initialize_params(w_init)
        self.dropout_rate = dropout_rate
        self.is_training = False  # Dropout only active when True

        # Safety check for SoftMax position
        for i, func in enumerate(self.a_functions):
            if func.__name__ == "SoftMax" and i != self.num_layers - 2:
                raise NetworkConfigError("SoftMax can only be used in the output-layer!")
            
    def train_mode(self):
        """Enable training mode (e.g., turn Dropout ON)."""
        self.is_training = True

    def eval_mode(self):
        """Enable evaluation mode (e.g., turn Dropout OFF)."""
        self.is_training = False

    def initialize_params(self, method: str):
        weights = []
        biases = [np.random.randn(y) * 0.1 for y in self.sizes[1:]]

        for x, y in zip(self.sizes[:-1], self.sizes[1:]):
            if method.lower() == "xavier":
                w = np.random.randn(x, y) * np.sqrt(1 / x)
            elif method.lower() == "he":
                w = np.random.randn(x, y) * np.sqrt(2 / x)
            elif method.lower() == "small":
                w = np.random.randn(x, y) / np.sqrt(y)
            else: # "large"
                w = np.random.randn(x, y)
            weights.append(w)
        
        return weights, biases

    def initialize_activations(self, a):
        if len(a) == self.num_layers - 1:
            return a
        raise LayerConfigError(f"Expected {self.num_layers - 1} activation functions, got {len(a)}.")

    def feedforward(self, a):
        """Passes input 'a' through the network with shape validation."""
        if a.shape[-1] != self.sizes[0]:
            raise ValueError(f"Input shape {a.shape} does not match the expected input size {self.sizes[0]}.")
    
        for i in range(self.num_layers - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.a_functions[i].forward(z)
        return a

    def __call__(self, a):
        """Allows calling the network like a function: output = net(input)"""
        return self.feedforward(a)

    def train(self, epochs: int, training_dataset: object, optimizer: object, 
              step_callback: Optional[Callable] = None, 
              epoch_callback: Optional[Callable] = None):
        
        if not hasattr(optimizer, 'NETWORK') or optimizer.NETWORK is None:
            optimizer.NETWORK = self

        start_time = time.time()
        print("-" * 30)
        print(f"Training started 🚀")
        print("-" * 30)

        persistent_metrics = {}

        try:
            for j in range(epochs):
                total_loss = 0
                
                if epoch_callback:
                    self.eval_mode() # Validation logic usually happens here
                    result = epoch_callback(epoch=j)
                    if isinstance(result, dict):
                        persistent_metrics.update(result)

                self.train_mode() # Turn Dropout ON for the training batch
                pbar = tqdm(training_dataset, desc=f"Epoch {j+1}/{epochs}", unit="batch")
                
                for index, (x, y) in enumerate(pbar):
                    loss = optimizer.step(x, y)
                    total_loss += loss
                    avg_loss = total_loss / (index + 1)
                    
                    if index % 5 == 0 or (index + 1) == len(training_dataset):
                        display_metrics = {"avg_loss": f"{avg_loss:.4f}"}
                        display_metrics.update(persistent_metrics)

                        if step_callback:
                            step_logs = step_callback(index=index, epoch=j)
                            if isinstance(step_logs, dict):
                                display_metrics.update(step_logs)
                        
                        pbar.set_postfix(display_metrics)
        finally:
            self.eval_mode() # Always reset to eval mode after training

        print("-" * 30)
        print(f"Finished in {round(time.time() - start_time, 2)}s 🥵")

    def save(self, filename: str):
        """Saves the network weights, biases, and structure to a JSON file."""
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            # We save the __name__ to make loading easier
            "activation_functions": [func.__class__.__name__ for func in self.a_functions],
            "dropout_rate": self.dropout_rate
        }
        with open(filename, "w") as f:
            json.dump(data, f)
        print(f"Model saved to {filename}!")

def load_from_file(filename: str):
    """Loads a network from a JSON file using standard Python getattr."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No file found at {filename}")

    with open(filename, "r") as f:
        data = json.load(f)
    
    a_funcs = []
    for name in data["activation_functions"]:
        # Get the class from our activationFunction module
        cls = getattr(af, name)
        a_funcs.append(cls())

    net = Network(data["sizes"], a_funcs, dropout_rate=data.get("dropout_rate", 0.0))
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    
    print("Network loaded successfully 🚀")
    return net