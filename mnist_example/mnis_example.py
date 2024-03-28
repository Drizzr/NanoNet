from NanoNet.network import Network, load_from_file
from NanoNet.optimizer import SGD, SGD_Momentum, RMSPromp, ADAM
from NanoNet.costFunction import QuadraticCost, CrossEntropy, LogLikelihood, CategorialCrossEntropy
from NanoNet.activationFunction import Sigmoid, ReLu, SoftMax
from NanoNet.data.examples import MNIST_DataSet_PKL
from NanoNet.data import DataLoader
import numpy as np


training_data = MNIST_DataSet_PKL('mnist_example/data/mnist.pkl.gz', type='train')
validation_data = MNIST_DataSet_PKL('mnist_example/data/mnist.pkl.gz', type='validation')

training_loader = DataLoader(training_data, batch_size=10, shuffle=True, drop_last=True)


net = Network([784, 50, 10], [ReLu(), Sigmoid()])
cost_function = QuadraticCost(net, False, False)

def epoch_callback(epoch):
    cost = 0.0
    for x, y in training_data:

        a = net.feedforward(x)

        cost += cost_function.forward(a, y)/ len(training_data)
    
    print(f"Cost: {cost}")

    results = [(np.argmax(net.feedforward(x)), np.argmax(y))
                        for (x, y) in training_data]
    
    print(f"Validation accuracy: {sum(int(x == y) for (x, y) in results) / len(training_data) * 100}%")

optimizer = SGD_Momentum(net, cost_function, 0.5)
net.train(100, optimizer=optimizer, training_dataset=training_loader, epoch_callback = epoch_callback)

"""
net.save("data.json")


net = load_from_file(filename="mnist_example/example_params.json")

#print(net.feedforward(training_data[0][0], batch=False))
#net.train(10, plot=True, monitor_training_cost=False, monitor_training_accuracy=False, monitor_test_cost=True, monitor_test_accuracy=True, test_convert=True)

results = [(np.argmax(net.feedforward(x)), np.argmax(y))
                        for (x, y) in training_data]

print(f"Validation accuracy: {sum(int(x == y) for (x, y) in results) / len(training_data) * 100}%")"""