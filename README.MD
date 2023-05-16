# NanoNet
A custom built neural network libary

## Project Overview 
This is a personal project of mine, which was heavily inspired by 
[the free e-book of Aaron Courville](http://neuralnetworksanddeeplearning.com). It implements a standard dense neural network (no convolutional-layers)

In the example-folder you can see how to train the Network with a dataset (in this case the famous mnist-dataset).


In order to train the network you have to provide samples of data, which has to be stored in the following format:

Say you are using n-input paramters, then your trainig-data set has to be an array of tuples containig:
1. The trainig input (one-sample) in an array of the shape (n,)
2. The desired output value either in the form of a number or an unit-array with a 1 at the desired index
    (index 1 == 1)

If either your trainig-data or test-data isn't provided in the unit-vector format you have to set the convert argument to True!


## Activation Functions:
The available activation-functions are Sigmoid, ReLu and SoftMax. Note that SoftMax can only be used in the output-layer in combination with either the Loglikelihood or the CategorialCrossEntropy-Cost-Function.
Using ReLu in the output-layer is almost never a good idea, since it messes with the logarythmic-based cost-functions (not nummerical stable).


## Cost Functions:
You choose between the CategorialCrossEntropy, CrossEntropy, LogLikelihood, Qudratic-Cost and MeanAbsoluteCost-Functions.
Note that certain cost-functions can only be used with certain activations-functions in the output layer.


## Optimizer:
You can choose between the most common optimizer such as ADAM, RMS_PROMP, SGD_MOMENTUM and normal SGD!


## Regression
The standard option is a Classifier type of Neural Network! If you want to use the Network for regression you have to set the classify argument to False. You can provide a custom precission threshhold in order to check for trainig accuaracy.
Note that you can only use the Qudratic-Cost and MeanAbsoluteCost-Functions for regression problems.

