import numpy as np

class LayerDense:
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, bias_regularizer_l1=0,
                 weight_regularizer_l2=0, bias_regularizer_l2=0):
        # initialize weights and biases
        
        # shape(inputs, neurons) so that we dont need to transpose every time
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        
        # shape(1, neurons)
        self.biases = np.zeros((1, n_neurons))
        
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l2 = bias_regularizer_l2
    
    def forward(self, inputs):
        # calculate output values from inputs, weights, and biases
        # no need to transpose the weights since we initialized it transposed
        self.output = np.dot(inputs, self.weights) + self.biases
        
        # store the inputs for the partial derivative during the backwards pass
        self.inputs = inputs
        
    def backward(self, dvalues):
        # gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # L1 regularization
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
            
        # L2 regularization
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # gradients on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        
        
class LayerDropout:
    def __init__(self, rate):
        # store the rate of what we want to keep
        self.rate = 1 - rate
        
    def forward(self, inputs):
        self.inputs = inputs
        
        # compute the scaled mask (and store for backwards pass)
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        
        # filter out the ouputs using the binary mask
        self.output = inputs * self.binary_mask
        
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask