from NanoNet.network import Network, load_from_file
from NanoNet.optimizer import SGD, SGD_Momentum, RMSPromp, ADAM
from NanoNet.costFunction import QuadraticCost, CrossEntropy, LogLikelihood
from NanoNet.activationFunction import Sigmoid, ReLu, SoftMax
from mnis_loadaer import load_data_wrapper

training_data, validation_data, test_data = load_data_wrapper()

#net = Network([784, 100, 20, 10], [ReLu(), ReLu(), SoftMax()], LogLikelihood(), ADAM(0.0002), 10, test_data=test_data, training_data=training_data, w_init_size="large")
#net.train(1, monitor_training_cost=False, monitor_training_accuracy=False, monitor_test_cost=True, monitor_test_accuracy=True)

#net.save("data.json")


net = load_from_file("data.json", training_data, test_data)

print(net.evaluate(training_data, convert=False)/len(training_data))