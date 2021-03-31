import numpy as np

class Loss:  
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
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
            
        # mask value - only for on-hot encoded labels
        elif len(y_true.shape) == 2:
            confidences = np.sum(y_pred_clipped * y_true, axis=1)
            
        return -np.log(confidences)
