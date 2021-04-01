import numpy as np

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        # initialize weights and biases
        
        # shape(inputs, neurons) so that we dont need to transpose every time
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        
        # shape(1, neurons)
        self.biases = np.zeros((1, n_neurons))        
    
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
        
        # gradients on values
        self.dinputs = np.dot(dvalues, self.weights.T)