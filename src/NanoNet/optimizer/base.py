import numpy as np
from abc import ABC, abstractmethod
from NanoNet.Exceptions import NetworkConfigError

class Optimizer(ABC):
    
    def __init__(self, network: object, cost_function: object) -> None:
        self.NETWORK = network
        self.COST_FUNCTION = cost_function

        # Ensure the Network and Optimizer are linked
        if hasattr(network, 'optimizer'):
            network.optimizer = self

        # Logic for Configuration Validation
        # Softmax requires CCE or LogLikelihood
        output_activation = self.NETWORK.a_functions[-1].__name__
        cost_name = self.COST_FUNCTION.__name__

        if cost_name in ["CategorialCrossEntropy", "LogLikelihood"]:
            if output_activation != "SoftMax":
                raise NetworkConfigError(
                    f"{cost_name} requires a SoftMax output layer, but found {output_activation}."
                )

        if cost_name == "CrossEntropy": # This matches your BinaryCrossEntropy.__name__
            if output_activation != "Sigmoid":
                raise NetworkConfigError(
                    f"BinaryCrossEntropy requires a Sigmoid output layer, but found {output_activation}."
                )
        
        if output_activation == "SoftMax" and cost_name not in ["CategorialCrossEntropy", "LogLikelihood"]:
             raise NetworkConfigError(
                "SoftMax activation can only be used with LogLikelihood or CategorialCrossEntropy."
            )

    def backprop(self, x, y):
        batch_size = x.shape[0]
        
        # 1. Forward Pass
        activations = [x] 
        zs = [] 
        masks = [] # Store dropout masks

        for i in range(self.NETWORK.num_layers - 1):
            z = np.dot(activations[-1], self.NETWORK.weights[i]) + self.NETWORK.biases[i]
            zs.append(z)
            
            activation = self.NETWORK.a_functions[i].forward(z)

            # --- DROPOUT FORWARD ---
            # Only apply if training, dropout_rate > 0, and NOT the output layer
            if self.NETWORK.is_training and self.NETWORK.dropout_rate > 0 and i < self.NETWORK.num_layers - 2:
                p = self.NETWORK.dropout_rate
                mask = (np.random.rand(*activation.shape) > p) / (1.0 - p)
                activation *= mask
                masks.append(mask)
            else:
                masks.append(None)

            activations.append(activation)

        # 2. Calculate Loss (Cost Function already handles L1/L2 penalty in its .forward)
        batch_loss = self.COST_FUNCTION.forward(activations[-1], y)

        # 3. Backward Pass
        nabla_b = [None] * len(self.NETWORK.biases)
        nabla_w = [None] * len(self.NETWORK.weights)

        # Output layer delta
        delta = self.COST_FUNCTION.delta(zs[-1], activations[-1], y, self.NETWORK.a_functions[-1])
        
        nabla_b[-1] = np.sum(delta, axis=0) 
        nabla_w[-1] = np.dot(activations[-2].T, delta)

        # Backpropagate through hidden layers
        for l in range(2, self.NETWORK.num_layers):
            z = zs[-l]
            sp = self.NETWORK.a_functions[-l].derivative(z)
            
            delta = np.dot(delta, self.NETWORK.weights[-l+1].T) * sp
            
            # --- DROPOUT BACKWARD ---
            # If a neuron was silenced forward, we silence the gradient backward
            if masks[-l] is not None:
                delta *= masks[-l]

            nabla_b[-l] = np.sum(delta, axis=0)
            nabla_w[-l] = np.dot(activations[-l-1].T, delta)

        # 4. REGULARIZATION (The part we can't forget!)
        # Regularization is independent of Dropout masks
        if self.COST_FUNCTION.l2 or self.COST_FUNCTION.l1:
            lambd = self.COST_FUNCTION.lambd
            for i in range(len(nabla_w)):
                if self.COST_FUNCTION.l2:
                    # L2 Gradient: lambda * weight
                    nabla_w[i] += lambd * self.NETWORK.weights[i]
                elif self.COST_FUNCTION.l1:
                    # L1 Gradient: lambda * sign(weight)
                    nabla_w[i] += lambd * np.sign(self.NETWORK.weights[i])

        return nabla_b, nabla_w, batch_loss

    def step(self, x, y):
        """
        Updates the network parameters and returns the loss for this batch.
        """
        return self.update_mini_batch(x, y)

    @abstractmethod
    def update_mini_batch(self, x, y):
        """Subclasses (SGD, ADAM, etc.) must implement this."""
        pass