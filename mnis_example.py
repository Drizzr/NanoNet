from NanoNet.network import Network
from NanoNet.optimizer import SGD
from NanoNet.costFunction import QuadraticCost
from NanoNet.activationFunction import Sigmoid
from mnis_loadaer import load_data_wrapper

training_data, validation_data, test_data = load_data_wrapper()

net = Network([784, 100, 10], SGD(30, 3),[Sigmoid(), Sigmoid()], QuadraticCost(), test_data=test_data, training_data=training_data)
net.train(3)