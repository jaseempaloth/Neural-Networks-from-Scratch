import numpy as np

class Activation:
    def forward(self, input_data):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError

class ReLU(Activation):
    def forward(self, input_data):
        self.mask = input_data > 0
        return input_data * self.mask
    
    def backward(self, grad_output):
        return grad_output * self.mask

class Sigmoid(Activation):
    def forward(self, input_data):
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output
    
    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)

class Tanh(Activation):
    def forward(self, input_data):
        self.output = np.tanh(input_data)
        return self.output

    def backward(self, grad_output):
        return grad_output * (1 - self.output ** 2)
    
class Softmax(Activation):
    def forward(self, input_data):
        exp_x = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.output = exp_x / exp_x.sum(axis=1, keepdims=True)
        return self.output
    
    def backward(self, grad_output):
        """
        Proper softmax gradient calculation using Jacobian matrix properties.
        Assumes grad_output is the gradient from the loss function.
        """
        batch_size = grad_output.shape[0]
        grad = np.empty_like(grad_output)

        for i in range(batch_size):
            single_output = self.output[i].reshape(-1, 1)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            grad[i] = np.dot(jacobian, grad_output[i])
        
        return grad

        
        


        
     

    