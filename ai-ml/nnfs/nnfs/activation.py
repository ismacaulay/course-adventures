import numpy as np
from loss import CategoricalCrossEntropy

class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        # store the inputs for the backwars pass
        self.inputs = inputs
        
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # zero gradient where input values are negative
        self.dinputs[self.inputs <= 0] = 0
        
class ActivationSoftmax:
    def forward(self, inputs):
        # get non-normalized probabilities
        # subtract the map value of the inputs to prevent exploding values
        # when dealing with really large numbers. it does not take much for
        # exponential values to get very large (ie. e^1000 will overflow)
        # since e^-inf == 0 and e^0 = 1, we subtract the largest value to put
        # the input values in the range of -inf to 0 (since largest itself == 0)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        # normalize
        # axis=1 means sum the rows, keepdims=True will keep it in the same dims
        # as the exp_values (ie. wont flatten to a list)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities
        
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # flatten out array
            single_output = single_output.reshape(-1, 1)
            
            # compute jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            
            # calculate the samplewise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
            

# Softmax Classifier - combined Softmax activation and cross-entropy loss
#                      for faster backwards step
class ActivationSoftmaxLossCategoricalCrossentropy:
    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = CategoricalCrossEntropy()
        
    def forward(self, inputs, y_true):
        # compute output
        self.activation.forward(inputs)
        self.output = self.activation.output
    
        # compute loss
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)
        
        # if labels are one-hot encoded, turn them
        # into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
            
        # compute gradient
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        
        # normalize
        self.dinputs = self.dinputs / samples