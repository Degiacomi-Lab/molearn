import os
import sys
import numpy as np
import torch
import random
import string


def random_string(length=32):
    '''
    generate a random string of arbitrary characters. Useful to generate temporary file names.

    :param length: length of random string
    '''
    return ''.join(random.choice(string.ascii_letters)
                    for n in range(length))


def as_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.data.cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        return np.array(tensor)
    

class ShutUp:
    
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, *args):
        sys.stdout.close()
        sys.stdout = self._stdout
        

class CheckpointBatchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, run_function, backward_batch_size, forward_batch_size):
        ctx.run_function = run_function
        ctx.save_for_backward(x)
        ctx.backward_batch_size = backward_batch_size

        with torch.no_grad():
            first_batch = run_function(x[:forward_batch_size])
            #empty = torch.empty_like(first_batch[0],shape=[x.shape[0]]+list(first_batch.shape[1:]))
            empty = torch.empty([x.shape[0]]+list(first_batch.shape[1:]),dtype=first_batch.dtype, device=first_batch.device)
            empty[:forward_batch_size] = first_batch
            for i in range(forward_batch_size, x.shape[0], forward_batch_size):
                empty[i:i+forward_batch_size] = run_function(x[i:i+forward_batch_size])
        return empty
    @staticmethod
    def backward(ctx, grad):
        input = ctx.saved_tensors[0]

        #detach 
        detached_input = input.detach()
        detached_input.requires_grad = input.requires_grad
        for i in range(0,detached_input.shape[0],ctx.backward_batch_size):
            with torch.enable_grad():
                output = ctx.run_function(detached_input[i:i+ctx.backward_batch_size])
            torch.autograd.backward([output],[grad[i:i+ctx.backward_batch_size]])

        return detached_input.grad, None, None, None
    
class CheckpointBatch(torch.nn.Module):
    def __init__(self, function, backward_batch_size = 2, forward_batch_size = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.function = function
        self.backward_batch_size = backward_batch_size
        self.forward_batch_size = forward_batch_size

    def forward(self, x):
        return CheckpointBatchFunction.apply(x.requires_grad_(), self.function, self.backward_batch_size, self.forward_batch_size)

