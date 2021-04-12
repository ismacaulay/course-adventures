import numpy as np
import nnfs
from layer import LayerDense, LayerDropout
from activation import ActivationReLU, ActivationSoftmax, ActivationSoftmaxLossCategoricalCrossentropy
from optimizer import OptimizerSGD, OptimizerAdaGrad, OptimizerRMSProp, OptimizerAdam
from loss import CategoricalCrossEntropy
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

# create dataset
X, y = spiral_data(samples=1000, classes=3)

# create a dense layer of 2 inputs, and 64 outputs
dense1 = LayerDense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)

# create a ReLU activation function to be used by the hidden layers
activation1 = ActivationReLU()

dropout1 = LayerDropout(0.1)

# create a dense layer of 64 inputs and 3 outputs
dense2 = LayerDense(512, 3)

loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()
# optimizer = OptimizerSGD(decay=1e-3, momentum=0.9)
# optimizer = OptimizerAdaGrad(decay=1e-4)
# optimizer = OptimizerRMSProp(learning_rate=0.02, decay=1e-5, rho=0.999)

# optimizer = OptimizerAdam(learning_rate=0.02, decay=1e-5) # acc: 0.967, loss: 0.081
optimizer = OptimizerAdam(learning_rate=0.05, decay=5e-5) # acc 0.967, loss: 0.074

for epoch in range(10001):
    # forward pass through layer 1
    dense1.forward(X)

    # forward pass of the activation function using the output of the first layer
    activation1.forward(dense1.output)

    # forward pass through the dropout layer
    dropout1.forward(activation1.output)
    
    # forward pass of the second layer using the output of the activation function
    dense2.forward(dropout1.output)

    # compute loss
    data_loss = loss_activation.forward(dense2.output, y)
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + \
                          loss_activation.loss.regularization_loss(dense2)
    loss = data_loss + regularization_loss
    
    # compute accuracy
    # predictions = np.argmax(softmax.output, axis=1)
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    
    if epoch % 100 == 0:
        print(f'epoch: {epoch}, accuracy: {accuracy:.3f}, ' + \
              f'loss: {loss:.3f}, data_loss: {data_loss:.3f}, reg_loss: {regularization_loss:.3f}, ' + \
              f'lr: {optimizer.current_learning_rate}')

    # backwards pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    # update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# validate the model
# create test data
X_test, y_test = spiral_data(samples=100, classes=3)

# forward pass through out model
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_test)

print(f'validation: accuracy: {accuracy:.3f}, loss: {loss:.3f}')

