from NanoNet.network import Network
from NanoNet.optimizer import SGD, SGD_Momentum
from NanoNet.costFunction import QuadraticCost, CrossEntropy, LogLikelihood
from NanoNet.activationFunction import Sigmoid, ReLu, SoftMax
from mnis_loadaer import load_data_wrapper

training_data, validation_data, test_data = load_data_wrapper()

net = Network([784, 100, 10], 10, SGD_Momentum(2),[Sigmoid(), SoftMax()], LogLikelihood(), test_data=test_data, training_data=training_data, w_init_size="large")
net.train(40, monitor_training_cost=False, monitor_training_accuracy=False, monitor_test_cost=True, monitor_test_accuracy=True)
