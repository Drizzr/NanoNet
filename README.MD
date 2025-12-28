

# NanoNet 🚀

NanoNet is a lightweight, professional-grade dense neural network library built from scratch in Python. It features a modular architecture inspired by modern frameworks like PyTorch, providing a clean API for research and educational purposes.

## Key Features

- **Advanced Optimizers**: Includes `ADAM` (with bias correction), `RMSProp`, `SGD with Momentum`, and standard `SGD`.
- **Numerical Stability**: Hardened against `NaN` and `Inf` errors using max-subtraction (Softmax) and epsilon-clipping (Log/Sigmoid).
- **Regularization & Dropout**: Integrated **Inverted Dropout** and **L1/L2 Regularization** to prevent overfitting.
- **Smart Initialization**: Supports `Xavier` and `He` weight initialization for faster convergence.
- **Professional UI**: Live training progress bars powered by `tqdm`, tracking running average loss and custom metrics.
- **Flexible Data Handling**: PyTorch-style `Dataset` and `DataLoader` classes with support for shuffling and mini-batching.

---

## Installation

Install directly from GitHub using `pip`:

```bash
# Install the latest stable version
pip install git+https://github.com/Drizzr/NanoNet.git@latest
```

---

## Quick Start: The MNIST Demo

NanoNet comes with a built-in command-line tool to see it in action immediately:

```bash
nanonet-mnist
```
This command trains a 3-layer network on the MNIST dataset, reaching ~95% accuracy in just a few epochs.

---

## Documentation

### 1. Creating a Network
A network is defined by a list of layer sizes and a list of activation function instances.

```python
from NanoNet import Network, ReLU, SoftMax

# 784 inputs (MNIST), 100 hidden neurons, 10 outputs
# Uses He initialization by default for ReLU
net = Network(
    sizes=[784, 100, 10], 
    a_functions=[ReLU(), SoftMax()],
    w_init="he",
    dropout_rate=0.2  # 20% Dropout on hidden layers
)
```

### 2. Supported Components

| Category | Options |
| :--- | :--- |
| **Activations** | `ReLU`, `Sigmoid`, `SoftMax` |
| **Optimizers** | `ADAM`, `RMSProp`, `SGD_Momentum`, `SGD` |
| **Cost Functions** | `CategorialCrossEntropy`, `BinaryCrossEntropy`, `LogLikelihood`, `QuadraticCost`, `MeanAbsoluteCost` |

### 3. Training with Live Feedback
Training supports custom callbacks that return dictionaries. These metrics are displayed live in the progress bar.

```python
from NanoNet import ADAM, CategorialCrossEntropy, DataLoader

# 1. Setup Optimizer & Loss
cost = CategorialCrossEntropy(net, l2=True, lambd=0.01)
optimizer = ADAM(net, cost, eta=0.001)

# 2. Setup Data
train_loader = DataLoader(my_dataset, batch_size=32, shuffle=True)

# 3. Define a metric callback
def val_callback(epoch):
    # Perform validation...
    return {"val_acc": "94.2%"}

# 4. Train
net.train(
    epochs=10, 
    training_dataset=train_loader, 
    optimizer=optimizer,
    epoch_callback=val_callback
)
```

### 4. Saving and Loading
Models are saved in a compact JSON format containing the architecture, weights, biases, and metadata.

```python
from NanoNet import load_from_file

# Save weights
net.save("my_model.json")

# Restore later
new_net = load_from_file("my_model.json")

# Inference
prediction = new_net(input_data) # Uses __call__ shortcut
```

---

## Technical Details

### Inverted Dropout
NanoNet uses **Inverted Dropout**. This means the weights are automatically scaled during training so that during inference (`feedforward`), no extra math is required. Simply call `net.eval_mode()` (done automatically after `train()`) to ensure all neurons are active for predictions.

### Numerical Stability
The library implements `np.clip` on all logarithmic inputs to prevent `log(0)` errors and uses the `max-subtraction` trick in the SoftMax layer to prevent `exp(z)` from overflowing memory.

---

## Acknowledgments
Heavily inspired by the mathematical foundations found in Aaron Courville's Deep Learning textbook.

---

## License
MIT License. See `LICENSE` for details.