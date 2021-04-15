import numpy as np

class Loss:  
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
    def regularization_loss(self, layer):
        regularization_loss = 0
        
        # L1 regularization
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
    
        # L2 regularization
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * \
                                   np.sum(np.abs(layer.weights * layer.weights))
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * \
                                   np.sum(np.abs(layer.biases * layer.biases))
            
        return regularization_loss
    
class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        
        # clip the data to prevent division by 0
        # clip both sides to no drag the mean towards any value
        # clip close to 0 and close to 1
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        # probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            confidences = y_pred_clipped[range(samples), y_true]
            
        # mask value - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            confidences = np.sum(y_pred_clipped * y_true, axis=1)
            
        return -np.log(confidences)
    
    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)
        
        # number of labels in each sample
        labels = len(dvalues[0])
        
        # if the labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        # compute gradient
        self.dinputs = -y_true / dvalues
        # normalize
        self.dinputs = self.dinputs / samples 
        

class BinaryCrossEntropy(Loss):
    
    def forward(self, y_pred, y_true):
        # clip the data to prevent division by 0
        # clip both sides to no drag the mean towards any value
        # clip close to 0 and close to 1
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        sample_losses = -(y_true * np.log(y_pred_clipped) + \
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        
        return sample_losses
        
    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)
        
        # number of outputs in every sample
        outputs = len(dvalues[0])
        
        # clip data to prevent division by 0
        # clip both sides to no drag the mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1-1e-7)
        
        # compute the gradient
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        
        # normalize
        self.dinputs = self.dinputs / samples