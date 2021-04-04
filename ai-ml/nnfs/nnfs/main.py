import numpy as np
import nnfs
from layer import LayerDense
from activation import ActivationReLU, ActivationSoftmax, ActivationSoftmaxLossCategoricalCrossentropy
from optimizer import OptimizerSGD
from loss import CategoricalCrossEntropy
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

# create dataset
X, y = spiral_data(samples=100, classes=3)

# create a dense layer of 2 inputs, and 64 outputs
dense1 = LayerDense(2, 64)

# create a ReLU activation function to be used by the hidden layers
activation1 = ActivationReLU()

# create a dense layer of 64 inputs and 3 outputs
dense2 = LayerDense(64, 3)

loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()
optimizer = OptimizerSGD(decay=1e-3, momentum=0.9)

for epoch in range(10001):
    # forward pass through layer 1
    dense1.forward(X)

    # forward pass of the activation function using the output of the first layer
    activation1.forward(dense1.output)

    # forward pass of the second layer using the output of the activation function
    dense2.forward(activation1.output)

    # compute loss
    loss = loss_activation.forward(dense2.output, y)

    # compute accuracy
    # predictions = np.argmax(softmax.output, axis=1)
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    
    if epoch % 100 == 0:
        print(f'epoch: {epoch}, accuracy: {accuracy:.3f}, ' + \
              f'loss: {loss:.3f}, lr: {optimizer.current_learning_rate}')

    # backwards pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
