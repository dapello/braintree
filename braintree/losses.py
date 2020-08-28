import torch
from torch import Tensor
from torch.nn import Module

class CenteredKernelAlignment(Module):
    
    def __init__(self):
        super(CenteredKernelAlignment, self).__init__()
            
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.shape[0] == target.shape[0]
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)
        return 1 - CKA_(input, target)

def frobdot(X: Tensor, Y: Tensor) -> Tensor:
    return torch.norm(torch.matmul(Y.t(), X), p='fro')
    
def CKA_(X: Tensor, Y: Tensor) -> Tensor:
    return frobdot(X,Y)**2 / (frobdot(X,X)*frobdot(Y,Y))
