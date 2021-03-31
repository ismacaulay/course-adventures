import numpy as np
import nnfs
from layer import LayerDense
from activation import ActivationReLU, ActivationSoftmax
from loss import CategoricalCrossEntropy
from nnfs.datasets import spiral_data


import matplotlib.pyplot as plt


nnfs.init()

# create a dense layer of 2 inputs, and 3 outputs
dense1 = LayerDense(2, 3)

# create a dense layer of 3 inputs and 3 outputs
dense2 = LayerDense(3, 3)

# create a ReLU activation function to be used by the hidden layers
activation1 = ActivationReLU()

# create a softmax activation function to be used by the output layer
softmax = ActivationSoftmax()

# create a loss function
loss_function = CategoricalCrossEntropy()

# create dataset
X, y = spiral_data(samples=100, classes=3)

# forward pass through layer 1
dense1.forward(X)

# forward pass of the activation function using the output of the first layer
activation1.forward(dense1.output)

# forward pass of the second layer using the output of the activation function
dense2.forward(activation1.output)

# forward pass of the softmax activation using the output of the second layer
softmax.forward(dense2.output)

# print some outputs
print(softmax.output[:5])

# compute loss
loss = loss_function.calculate(softmax.output, y)
print(f'loss: {loss}')

# compute accuracy
predictions = np.argmax(softmax.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

print(f'accuracy: {accuracy}')
