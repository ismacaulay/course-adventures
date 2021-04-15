import numpy as np
import nnfs
from layer import LayerDense
from activation import ActivationReLU, ActivationSigmoid
from optimizer import OptimizerAdam
from loss import BinaryCrossEntropy
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

# create dataset
X, y = spiral_data(samples=100, classes=2)

# reshape the labels to be a list of lists
y = y.reshape(-1, 1)

dense1 = LayerDense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation1 = ActivationReLU()
dense2 = LayerDense(64, 1)
activation2 = ActivationSigmoid()
loss_function = BinaryCrossEntropy()
optimizer = OptimizerAdam(decay=5e-7)

for epoch in range(10001):
    # forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)    
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    # compute loss
    data_loss = loss_function.calculate(activation2.output, y)
    regularization_loss = loss_function.regularization_loss(dense1) + \
                          loss_function.regularization_loss(dense2)
    loss = data_loss + regularization_loss
    
    # compute accuracy
    # we want predictions to be 0 or 1. this will convert it to True/False,
    # then multiplying by 1 gives us 0 or 1
    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions == y)
    
    if epoch % 100 == 0:
        print(f'epoch: {epoch}, accuracy: {accuracy:.3f}, ' + \
              f'loss: {loss:.3f}, data_loss: {data_loss:.3f}, reg_loss: {regularization_loss:.3f}, ' + \
              f'lr: {optimizer.current_learning_rate}')

    # backwards pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# validate the model
# create test data
X_test, y_test = spiral_data(samples=100, classes=2)
y_test = y_test.reshape(-1, 1)

# forward pass through out model
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss = loss_function.calculate(activation2.output, y_test)

predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions == y_test)

print(f'validation: accuracy: {accuracy:.3f}, loss: {loss:.3f}')

