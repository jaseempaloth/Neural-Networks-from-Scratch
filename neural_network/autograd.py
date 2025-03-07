import numpy as np
from typing import Callable, Union, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from tensor import Tensor

# Import the Tensor class indirectly to avoid circular imports
def _import_tensor():
    from tensor import Tensor
    return Tensor

class Context:
    """
    A context that stores information for backward computation.
    
    Attributes:
        saved_tensors (list): Tensors whose values are required for gradient computation.
        saved_values (list): Additional values required for gradient computation.
        function (callable): The function that will compute gradients during backward pass.
    """
    def __init__(self, function: Callable):
        """
        Initialize a new Context.
        
        Args:
            function: The function that will compute gradients during backward pass.
        """
        self.saved_tensors = []
        self.saved_values = []
        self.function = function
    
    def save_for_backward(self, *tensors: Union['Tensor', Any]):
        """
        Save tensors needed for the backward pass.
        
        Args:
            *tensors: Tensors or values to save for the backward pass.
        """
        Tensor = _import_tensor()
        for tensor in tensors:
            if isinstance(tensor, Tensor):
                self.saved_tensors.append(tensor)
            else:
                self.saved_values.append(tensor)

def _wrap_numpy(value: Any) -> 'Tensor':
    """
    Wrap a numpy array, list, or scalar into a Tensor.
    
    Args:
        value: The value to wrap.
        
    Returns:
        Tensor: The wrapped value.
    """
    Tensor = _import_tensor()
    if isinstance(value, Tensor):
        return value
    return Tensor(value)

def apply_operation(function: Callable,
                    ctx_function: Callable,
                    *args: Any,
                    **kwargs: Any) -> 'Tensor':
    """
    Apply an operation and set up autograd information.
    
    Args:
        function: The forward function.
        ctx_function: The backward function.
        *args: Arguments to the function.
        **kwargs: Keyword arguments to the function.
        
    Returns:
        Tensor: The result of the operation.
    """
    Tensor = _import_tensor()
    args = [_wrap_numpy(arg) for arg in args]

    # Check if any of the input tensors require gradients
    requires_grad = any(t.requires_grad for t in args if isinstance(t, Tensor))

    # Compute forward pass
    result_data = function(*[t.data if isinstance(t, Tensor) else t for t in args], **kwargs)

    # If no gradient required, return a tensor without grad tracking
    if not requires_grad:
        return Tensor(result_data)
    
    # Create context for backward pass
    ctx = Context(ctx_function)
    ctx.save_for_backward(*args)

    # Return a tensor with gradient tracking
    return Tensor(result_data, requires_grad=True, ctx=ctx)

def backward(tensor: 'Tensor', grad: np.ndarray = None):
    """
    Perform a backward pass starting from the given tensor.
    
    Args:
        tensor: The tensor to start backpropagation from.
        grad: The gradient from upstream operations.
    """
    if tensor._ctx is None:
        return
    
    if grad is None:
        grad = np.ones_like(tensor.data)
    elif np.isscalar(grad):
        # Convert scalar to array
        grad = np.ones_like(tensor.data) * grad
    
    # Update the gradient
    tensor.grad = tensor.grad + grad

    # Compute gradients for inputs
    input_grads = tensor._ctx.function(tensor._ctx, grad)

    # Apply gradients to input tensors
    if not isinstance(input_grads, tuple):
        input_grads = (input_grads,)

    # Create a dictionary to accumulate gradients for each tensor
    tensor_to_grad = {}

    # First, accumulate all gradients
    for tensor_idx, input_grad in enumerate(input_grads):
        if input_grad is not None and tensor_idx < len(tensor._ctx.save_tensors):
            input_tensor = tensor._ctx.saved_tensors[tensor_idx]

            if input_tensor.requires_grad:
                if input_tensor in tensor_to_grad:
                    # Accumulate gradients for the same tensor
                    tensor_to_grad[input_tensor] = tensor_to_grad[input_tensor] + input_grad
                else:
                    tensor_to_grad[input_tensor] = input_grad
    
    # Then, apply accumulated gradients and continue backpropagation
    for input_tensor, accumulated_grad in tensor_to_grad.items():
        # Update the gradient directly
        input_tensor.grad = input_tensor.grad + accumulated_grad
        # Continue backpropagation if the tensor has a context
        if input_tensor._ctx is not None:
            backward(input_tensor, None)

# Operations and their gradients
def add(a, b):
    """Addition operation with autograd support."""
    def _forward(a_data, b_data):
        return a_data + b_data
    
    def _backward(ctx, grad):
        a, b = ctx.saved_tensors
        # Return gradients for both inputs
        return grad, grad
    
    return apply_operation(_forward, _backward, a, b)

def sub(a, b):
    """Subtraction operation with autograd support."""
    def _forward(a_data, b_data):
        return a_data - b_data
    
    def _backward(ctx, grad):
        a, b = ctx.saved_tensors
        return grad, -grad
    
    return apply_operation(_forward, _backward, a, b)

def mul(a, b):
    """Multiplication operation with autograd support."""
    def _forward(a_data, b_data):
        return a_data * b_data
    
    def _backward(ctx, grad):
        a, b = ctx.saved_tensors
        
        # Check if a and b are the same tensor (for x * x case)
        if a is b:
            # For x * x, the gradient is 2 * x * grad
            return 2 * a.data * grad, None
        
        # For regular multiplication, return gradients for both inputs
        return b.data * grad, a.data * grad
    
    return apply_operation(_forward, _backward, a, b)

def div(a, b):
    """Division operation with autograd support."""
    def _forward(a_data, b_data):
        return a_data / b_data
    
    def _backward(ctx, grad):
        a, b = ctx.saved_tensors
        return grad / b.data, -grad * a.data / (b.data * b.data)
    
    return apply_operation(_forward, _backward, a, b)

def matmul(a, b):
    """Matrix multiplication with autograd support."""
    def _forward(a_data, b_data):
        return np.matmul(a_data, b_data)
    
    def _backward(ctx, grad):
        a, b = ctx.saved_tensors
        return np.matmul(grad, b.data.T), np.matmul(a.data.T, grad)
    
    return apply_operation(_forward, _backward, a, b)

def pow(a, power):
    """Power operation with autograd support."""
    def _forward(a_data, power_val):
        # Ensure power_val is a scalar if it's a tensor
        if hasattr(power_val, 'data'):
            power_val = power_val.data
        return np.power(a_data, power_val)
    
    def _backward(ctx, grad):
        a, power_val = ctx.saved_tensors
        # Ensure power_val is a scalar if it's a tensor
        if hasattr(power_val, 'data'):
            power_val = power_val.data
        return grad * power_val * np.power(a.data, power_val - 1), None
    
    return apply_operation(_forward, _backward, a, power)

def neg(a):
    """Negation operation with autograd support."""
    def _forward(a_data):
        return -a_data
    
    def _backward(ctx, grad):
        return -grad
    
    return apply_operation(_forward, _backward, a)

def sum(a, axis=None, keepdims=False):
    """Sum operation with autograd support."""
    def _forward(a_data):
        return np.sum(a_data, axis=axis, keepdims=keepdims)
    
    def _backward(ctx, grad):
        a = ctx.saved_tensors[0]
        if not keepdims and axis is not None:
            grad_shape = list(a.data.shape)
            if isinstance(axis, (list, tuple)):
                for ax in axis:
                    grad_shape[ax] = 1
            else:
                grad_shape[axis] = 1
            grad = grad.reshape(grad_shape)
        return np.broadcast_to(grad, a.data.shape)
    
    return apply_operation(_forward, _backward, a)

def mean(a, axis=None, keepdims=False):
    """Mean operation with autograd support."""
    def _forward(a_data):
        return np.mean(a_data, axis=axis, keepdims=keepdims)
    
    def _backward(ctx, grad):
        a = ctx.saved_tensors[0]
        if not keepdims and axis is not None:
            grad_shape = list(a.data.shape)
            if isinstance(axis, (list, tuple)):
                for ax in axis:
                    grad_shape[ax] = 1
            else:
                grad_shape[axis] = 1
            grad = grad.reshape(grad_shape)
        
        if axis is None:
            n = np.prod(a.data.shape)
        else:
            n = np.prod([a.data.shape[i] for i in (axis if isinstance(axis, (list, tuple)) else [axis])])
        
        return np.broadcast_to(grad, a.data.shape) / n
    
    return apply_operation(_forward, _backward, a)

def exp(a):
    """Exponential operation with autograd support."""
    def _forward(a_data):
        return np.exp(a_data)
    
    def _backward(ctx, grad):
        result = ctx.saved_values[0]  # We saved the forward result
        return grad * result
    
    result = apply_operation(_forward, _backward, a)
    result._ctx.saved_values.append(result.data)  # Save forward result for backward
    return result

def log(a):
    """Natural logarithm operation with autograd support."""
    def _forward(a_data):
        return np.log(a_data)
    
    def _backward(ctx, grad):
        a = ctx.saved_tensors[0]
        return grad / a.data
    
    return apply_operation(_forward, _backward, a)

def reshape(a, shape):
    """Reshape operation with autograd support."""
    def _forward(a_data):
        return np.reshape(a_data, shape)
    
    def _backward(ctx, grad):
        a = ctx.saved_tensors[0]
        return np.reshape(grad, a.data.shape)
    
    return apply_operation(_forward, _backward, a)






        
        
        

