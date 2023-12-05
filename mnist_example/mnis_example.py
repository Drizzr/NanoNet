from NanoNet.network import Network, load_from_file
from NanoNet.optimizer import SGD, SGD_Momentum, RMSPromp, ADAM
from NanoNet.costFunction import QuadraticCost, CrossEntropy, LogLikelihood, CategorialCorssEntropy
from NanoNet.activationFunction import Sigmoid, ReLu, SoftMax
from mnis_loadaer import load_data_wrapper


#net.save("data.json")


#net = load_from_file("mnist_example/example_params.json", training_data, test_data)

#print(net.evaluate(training_data, convert=False)/len(training_data))
#print(net.evaluate(test_data)/len(test_data))