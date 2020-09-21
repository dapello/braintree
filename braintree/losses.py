import torch
from torch import Tensor
from torch.nn import Module

class CenteredKernelAlignment(Module):
    
    def __init__(self):
        super(CenteredKernelAlignment, self).__init__()
            
    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        assert X.shape[0] == Y.shape[0]
        X, Y = X.view(X.shape[0], -1), Y.view(Y.shape[0], -1)
        X = X - X.mean(dim=0)
        Y = Y - Y.mean(dim=0)
        return 1 - CKA_(X, Y)
    
def CKA_(X: Tensor, Y: Tensor) -> Tensor:
    return frobdot(X,Y)**2 / (frobdot(X,X)*frobdot(Y,Y))

def frobdot(X: Tensor, Y: Tensor) -> Tensor:
    return torch.norm(torch.matmul(Y.t(), X), p='fro')
