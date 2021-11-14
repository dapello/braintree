import torch as ch
from torch import Tensor
from torch.nn import Module


class LogCenteredKernelAlignment(Module):

    name = 'logCKA'

    def __init__(self, device='gpu'):
        super(LogCenteredKernelAlignment, self).__init__()
        self.device=device
            
    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        if self.device == 'gpu':
            X = X.cuda()
            Y = Y.cuda()
        elif self.device == 'cpu':
            X = X.cpu()
            Y = Y.cpu()

        assert X.shape[0] == Y.shape[0]
        X, Y = X.view(X.shape[0], -1), Y.view(Y.shape[0], -1)
        X = X - X.mean(dim=0)
        Y = Y - Y.mean(dim=0)
        return ch.log(1 - CKA(X, Y))

class CenteredKernelAlignment(Module):

    name = 'CKA'

    def __init__(self, device='gpu'):
        super(CenteredKernelAlignment, self).__init__()
        self.device=device
            
    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        if self.device == 'gpu':
            X = X.cuda()
            Y = Y.cuda()
        elif self.device == 'cpu':
            X = X.cpu()
            Y = Y.cpu()

        assert X.shape[0] == Y.shape[0]
        X, Y = X.view(X.shape[0], -1), Y.view(Y.shape[0], -1)
        X = X - X.mean(dim=0)
        Y = Y - Y.mean(dim=0)
        return 1 - CKA(X, Y)
    
def CKA(X: Tensor, Y: Tensor) -> Tensor:
    return frobdot(X,Y)**2 / (frobdot(X,X)*frobdot(Y,Y))

def frobdot(X: Tensor, Y: Tensor) -> Tensor:
    return ch.norm(ch.matmul(Y.t(), X), p='fro')

# alternate implementation:
def centering(K):
    n = K.shape[0]
    unit = ch.ones([n, n])
    I = ch.eye(n)
    H = I - unit / n

    return ch.dot(ch.dot(H, K), H)
    # HKH are the same with KH, KH is the first centering, H(KH) do the second time,
    # results are the sme with one time centering
    # return np.dot(H, K)  # KH


def linear_HSIC(X, Y):
    L_X = ch.dot(X, X.T)
    L_Y = ch.dot(Y, Y.T)
    return ch.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = ch.sqrt(linear_HSIC(X, X))
    var2 = ch.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)
