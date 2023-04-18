# NanoNet
A custom built neural network libary

## Project Overview 
This is a personal project of mine wich was heavily inspired by 
[the free e-book of Aaron Courville](http://neuralnetworksanddeeplearning.com). It implements a standard dense neural network (no convolutional-layers or what so ever)

In the example-folder you can see how the network can be trainied with a dataset (in this case the famous mnist-dataset)

In order to train the network you have to provide samples of data wich has to be stored in the following format:

Say you are using n-input paramters, then your trainig-data set has to be an array of tuples containig:
1. The trainig input (one-sample) in an array of the shape (n,)
2. The desired output value either in the form of a number or an unit-array with a 1 at the desired index
    (index 1 == 1)

If either your trainig-data or test-data isn't provided in the unit-vector format you have to set convert to True!


### Activation Functions:
The available activation-functions are Sigmoid, ReLu and SoftMax. Note that SoftMax can only be used in the output-layer in combination with either the Loglikelihood or the CategorialCrossEntropy-Cost-Function


### Cost Functions:
You choose between the CategorialCrossEntropy, CrossEntropy, LogLikelihood and the Qudratic-Cost-Functions.
Note that certain cost-functions can only be used with certain activations-functions in the output layer.


### Optimizer:
The Optimizer implemented are stantard Stochastic Gradient Descent, SGD-Momentum, ADAM and RMSpromp!

Note that the Libary currently only supports classification problems!