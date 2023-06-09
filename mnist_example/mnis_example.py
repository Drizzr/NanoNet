from NanoNet.network import Network, load_from_file
from NanoNet.optimizer import SGD, SGD_Momentum, RMSPromp, ADAM
from NanoNet.costFunction import QuadraticCost, CrossEntropy, LogLikelihood, CategorialCorssEntropy
from NanoNet.activationFunction import Sigmoid, ReLu, SoftMax
from mnis_loadaer import load_data_wrapper

training_data, validation_data, test_data = load_data_wrapper()

net = Network([784, 100, 10], [ReLu(), Sigmoid()], QuadraticCost(), SGD_Momentum(0.5), 10, 
              test_data=test_data, training_data=training_data, w_init_size="small")

net.train(10, plot=True, monitor_training_cost=False, monitor_training_accuracy=True, monitor_test_cost=True, monitor_test_accuracy=True, test_convert=True)

#net.save("data.json")


#net = load_from_file("mnist_example/example_params.json", training_data, test_data)

#print(net.evaluate(training_data, convert=False)/len(training_data))
#print(net.evaluate(test_data)/len(test_data))