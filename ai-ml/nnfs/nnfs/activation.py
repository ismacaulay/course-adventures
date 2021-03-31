import numpy as np

class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
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
        