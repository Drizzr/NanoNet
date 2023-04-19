# NanoNet
A custom built neural network libary

## Project Overview 
This is a personal project of mine wich was heavily inspired by 
[the free e-book of Aaron Courville](http://neuralnetworksanddeeplearning.com). It implements a standard dense neural network (no convolutional-layers)

In the example-folder you can see how to train the Network with a dataset (in this case the famous mnist-dataset).


In order to train the network you have to provide samples of data. which has to be stored in the following format:

Say you are using n-input paramters, then your trainig-data set has to be an array of tuples containig:
1. The trainig input (one-sample) in an array of the shape (n,)
2. The desired output value either in the form of a number or an unit-array with a 1 at the desired index
    (index 1 == 1)

If either your trainig-data or test-data isn't provided in the unit-vector format you have to set convert to True!


### Activation Functions:
The available activation-functions are Sigmoid, ReLu and SoftMax. Note that SoftMax can only be used in the output-layer in combination with either the Loglikelihood or the CategorialCrossEntropy-Cost-Function.
Using ReLu in the output-layer is almost never a good idea, since it messes with the logarythmic-based cost-functions.


### Cost Functions:
You choose between the CategorialCrossEntropy, CrossEntropy, LogLikelihood, Qudratic-Cost and MeanAbsoluteCost-Functions.
Note that certain cost-functions can only be used with certain activations-functions in the output layer.


### Optimizer:
The Optimizer implemented are stantard Stochastic Gradient Descent, SGD-Momentum, ADAM and RMSpromp!


## Regression
The standard option is a Classifier! If you want to use the Network for Regression you have to set the classify argument to False. You can provide a custom percission threshhold in order to check for trainig accuaracy.
Note that you can only use the Qudratic-Cost and MeanAbsoluteCost-Functions for Regression problems.

