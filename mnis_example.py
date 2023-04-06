from NanoNet.network import Network
from NanoNet.optimizer import SGD
from NanoNet.costFunction import QuadraticCost, CrossEntropy, LogLikelihood
from NanoNet.activationFunction import Sigmoid, ReLu, SoftMax
from mnis_loadaer import load_data_wrapper

training_data, validation_data, test_data = load_data_wrapper()

net = Network([784, 100, 10], SGD(10, 0.5, lamb=5),[ReLu(), SoftMax()], LogLikelihood(l2=True), test_data=test_data, training_data=training_data, w_init_size="large")
net.train(10, monitor_training_cost=False, monitor_training_accuracy=False, monitor_test_cost=True, monitor_test_accuracy=True)
