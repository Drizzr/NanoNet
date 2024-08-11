# NanoNet

This is a personal project of mine, heavily inspired by the free e-book by Aaron Courville. It implements a standard dense neural network (no convolutional layers).

## Quick Start

In the `mnist_example` folder, you can see how to train the network with a dataset (in this case, the famous MNIST dataset).

## Documentation

### Creating a New Neural Network

A new neural network can be defined as follows:

```python
from NanoNet.network import Network
net = Network([784, 50, 10], [Sigmoid(), SoftMax()])
```

The first array defines the size of each layer. Note that this library only supports simple feed-forward neural networks.

The second array specifies the activation functions (for n layers, you will need n-1 activation functions).

The neural network class also has the parameter `w_init_size` (default="small"), which initializes small weights if set accordingly.

### Activation Functions

This framework supports the usual activation functions:

- Sigmoid
- ReLU
- Softmax (Note: Softmax function may only be used in the output layer.)

### Cost/Loss Functions

You can choose between the following cost/loss functions:

- CategoricalCrossEntropy
- BinaryCrossEntropy
- LogLikelihood
- QuadraticCost
- MeanAbsoluteCost

Note: CategoricalCrossEntropy and LogLikelihood are essentially the same. Certain cost functions can only be used with specific activation functions in the output layer.

You can also specify whether you would like to use regularization (L1 or L2) along with the lambda value if chosen.

### Optimizers

You can choose between the most common optimizers:

- ADAM
- RMSProp
- SGD_Momentum
- SGD

You can specify their standard parameters.

### Training a Neural Network

After defining your neural network class, you will need to provide a dataset. You can use the custom dataset class as follows:

```python
class MNIST_DataSet_PKL(Dataset):
    def __init__(self, data_path: str, type: str = 'train'):
        super().__init__()
        self.data_path = data_path
        self.type = type
        self.data = self.load_data()

    def load_data(self):
        with gzip.open('mnist_example/data/mnist.pkl.gz', 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            training_data, validation_data, test_data = u.load()
        if self.type == 'train':
            return training_data
        elif self.type == 'validation':
            return validation_data
        elif self.type == 'test':
            return test_data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return self.data[0][idx], vectorized_result(self.data[1][idx], 10) # format: (784), (10)
```

(For a custom dataset, you will need to override the `__len__` and `__getitem__` methods. This is very similar to PyTorch by design and should be compatible. The libary also provides a dataloader similar to PyTorch)

Every neural network instance has a `train` method, which can be called as follows:

```python
net.train(100, optimizer=optimizer, training_dataset=training_loader, epoch_callback=epoch_callback)
```

The first integer specifies the number of epochs.

You can provide a custom step and/or epoch callback, for example, to track the loss or make backups.

You can save the network as follows:

```python
net.save("data.json")
```

(Note: Only JSON format is supported so far.)

You can also load from a checkpoint:

```python
net = load_from_file(filename="mnist_example/example_params.json")
```

If you want to infer some data, just call:

```python
net.feedforward(x)
```

For more details check out the mnist_example.py file!