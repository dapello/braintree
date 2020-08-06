import torch
from torch import Tensor
from torch.nn import Module

class CenteredKernelAlignment(Module):
    
    def __init__(self):
        super(CenterKernelAlignment, self).__init__()
    
            
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return 1 - CKA_(input, target)

def frobdot(X: Tensor, Y: Tensor) -> Tensor:
    return torch.norm(torch.matmul(Y.t(), X), p='fro')
    
def CKA_(X: Tensor, Y: Tensor, reduction) -> Tensor:
    return frobdot(X,Y)**2 / (frobdot(X,X)*frobdot(Y,Y))
