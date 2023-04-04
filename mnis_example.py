from NanoNet.network import Network
from NanoNet.optimizer import SGD
from NanoNet.costFunction import QuadraticCost, CrossEntropy, LogLikelihood
from NanoNet.activationFunction import Sigmoid, ReLu, SoftMax
from mnis_loadaer import load_data_wrapper

training_data, validation_data, test_data = load_data_wrapper()

net = Network([784, 20, 30, 10], SGD(30, 0.1),[Sigmoid(), Sigmoid(), SoftMax()], LogLikelihood(), test_data=test_data, training_data=training_data)

net.train(10)