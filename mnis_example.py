from NanoNet.network import Network
from NanoNet.optimizer import SGD
from NanoNet.costFunction import QuadraticCost, CrossEntropy, LogLikelihood
from NanoNet.activationFunction import Sigmoid, ReLu, SoftMax
from mnis_loadaer import load_data_wrapper

training_data, validation_data, test_data = load_data_wrapper()

net = Network([784, 100, 10], SGD(10, 3, lamb=100),[Sigmoid(), Sigmoid()], QuadraticCost(), test_data=test_data, training_data=training_data)

net.train(10, monitor_training_cost=True, monitor_training_accuracy=True, monitor_test_cost=False, monitor_test_accuracy=False)