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

# alternate implementation:
def centering(K):
    n = K.shape[0]
    unit = torch.ones([n, n])
    I = torch.eye(n)
    H = I - unit / n

    return torch.dot(torch.dot(H, K), H)
    # HKH are the same with KH, KH is the first centering, H(KH) do the second time,
    # results are the sme with one time centering
    # return np.dot(H, K)  # KH


def linear_HSIC(X, Y):
    L_X = torch.dot(X, X.T)
    L_Y = torch.dot(Y, Y.T)
    return torch.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    var2 = torch.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)
