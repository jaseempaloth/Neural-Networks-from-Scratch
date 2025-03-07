import numpy as np
from typing import Union, Optional, Tuple
import autograd

class Tensor:
    """
    A class that wraps numpy arrays and records operations for automatic differentiation.
    
    Attributes:
        data (np.ndarray): The actual data stored in the tensor.
        grad (np.ndarray): Gradient of the tensor with respect to some scalar.
        _requires_grad (bool): Whether to compute gradients for this tensor.
        _ctx (autograd.Context): The autograd context for gradient computation.
    """

    def __init__(self,
                data: Union[np.ndarray, list, float, int],
                requires_grad: bool = False,
                ctx: Optional['autograd.Context'] = None):
        """
        Initialize a new Tensor.
        
        Args:
            data: Data to be stored in the tensor.
            requires_grad: Whether to compute gradients for this tensor.
            ctx: The autograd context for gradient computation.
        """
        if isinstance(data, (np.ndarray, list, float, int)):
            self.data = np.array(data, dtype=np.float64)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._requires_grad = requires_grad
        self._ctx = ctx # or autograd.Context()

    @property
    def shape(self) -> Tuple:
        """Get the shape of the tensor."""
        return self.data.shape

    @property
    def requires_grad(self) -> bool:
        """Get whether the tensor requires gradients."""
        return self._requires_grad

    def __repr__(self) -> str:
        """String representation of the tensor."""
        return f"Tensor(data={self.data}, requires_grad={self._requires_grad})"
    
    # Basic arithmetic operations
    def __add__(self, other):
        """Add two tensors."""
        return autograd.add(self, other)
    
    def __sub__(self, other):
        """Subtract two tensors."""
        return autograd.sub(self, other)

    def __mul__(self, other):
        """Multiply two tensors."""
        return autograd.mul(self, other)
    
    def __truediv__(self, other):
        """Divide two tensors."""
        return autograd.div(self, other)
    
    def __matmul__(self, other):
        """Matrix multiply two tensors."""
        return autograd.matmul(self, other)
    
    def __pow__(self, power):
        """Raise self to the power of power."""
        return autograd.pow(self, power)
    
    def __neg__(self):
        """Negate the tensor."""
        return autograd.neg(self)

    # Reversed operations
    def __radd__(self, other):
        """Add self to other (reversed)."""
        return autograd.add(other, self)
    
    def __rsub__(self, other):
        """Subtract self from other (reversed)."""
        return autograd.sub(other, self)

    def __rmul__(self, other):
        """Multiply other by self (reversed)."""
        return autograd.mul(other, self)
    
    def __rtruediv__(self, other):
        """Divide other by self (reversed)."""
        return autograd.div(other, self)
    
    # Other operations
    def sum(self, axis=None, keepdims=False):
        """Sum the tensor along the specified axis."""
        return autograd.sum(self, axis=axis, keepdims=keepdims)
    
    def mean(self, axis=None, keepdims=False):
        """Mean of the tensor along the specified axis."""
        return autograd.mean(self, axis=axis, keepdims=keepdims)
    
    def exp(self):
        """Compute the exponential of the tensor."""
        return autograd.exp(self)
    
    def log(self):
        """Compute the natural logarithm of the tensor."""
        return autograd.log(self)
    
    def reshape(self, *shape):
        """Reshape the tensor."""
        return autograd.reshape(self, shape)
    
    def backward(self, grad: Optional[np.ndarray] = None):
        """
        Compute gradients of this tensor with respect to its inputs.
        
        Args:
            grad: Gradient from downstream operations. 
                  If None, it defaults to a tensor of ones.
        """
        if self._ctx is None:
            return 
        
        if grad is None:
            grad = np.ones_like(self.data)
        
        autograd.backward(self, grad)
    
    def zero_grad(self):
        """Zero out the gradient."""
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        

        





        

        
