import torch as ch
from torch import Tensor
from torch.nn import Module


class _CenteredKernelAlignment(Module):
    def __init__(self, fnc, name, device='gpu'):
        super(_CenteredKernelAlignment, self).__init__()
        self.fnc = fnc
        self.name = name
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
        #return ch.log(self.start - CKA(X, Y))
        return self.fnc(CKA(X, Y))

def LogCenteredKernelAlignment():
    def fnc(X):
        return ch.log(1 - X)
    return _CenteredKernelAlignment(fnc=fnc, name='logCKA')

def FlipLogCenteredKernelAlignment():
    def fnc(X):
        return -ch.log(1 - X)
    return _CenteredKernelAlignment(fnc=fnc, name='fliplogCKA')

def CenteredKernelAlignment():
    def fnc(X):
        return 1 - X
    return _CenteredKernelAlignment(fnc=fnc, name='CKA')

def FlipCenteredKernelAlignment():
    def fnc(X):
        return X
    return _CenteredKernelAlignment(fnc=fnc, name='flipCKA')

def LogCenteredKernelAlignment0():
    def fnc(X):
        return - ch.log(X)
    return _CenteredKernelAlignment(fnc=fnc, name='logCKA0')

def CKA(X: Tensor, Y: Tensor) -> Tensor:
    return frobdot(X,Y)**2 / (frobdot(X,X)*frobdot(Y,Y))

def frobdot(X: Tensor, Y: Tensor) -> Tensor:
    return ch.norm(ch.matmul(Y.t(), X), p='fro')

# alternate implementation:
class LogCenteredKernelAlignment2(Module):

    name = 'logCKA'

    def __init__(self, device='gpu'):
        super(LogCenteredKernelAlignment2, self).__init__()
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
        return ch.log(1 - linear_CKA(X, Y))

class CenteredKernelAlignment2(Module):

    name = 'CKA'

    def __init__(self, device='gpu'):
        super(CenteredKernelAlignment2, self).__init__()
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
        return 1 - linear_CKA(X, Y)

def centering(K):
    n = K.shape[0]
    unit = ch.ones([n, n])
    I = ch.eye(n)
    H = (I - unit / n).cuda()

    return ch.matmul(ch.matmul(H, K), H)
    # HKH are the same with KH, KH is the first centering, H(KH) do the second time,
    # results are the sme with one time centering
    # return np.dot(H, K)  # KH


def linear_HSIC(X, Y):
    L_X = ch.matmul(X, X.T)
    L_Y = ch.matmul(Y, Y.T)
    return ch.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = ch.sqrt(linear_HSIC(X, X))
    var2 = ch.sqrt(linear_HSIC(Y, Y))
    import pdb; pdb.set_trace()

    return hsic / (var1 * var2)



NEURAL_LOSSES = {
    'CKA' : CenteredKernelAlignment,
    'flipCKA' : FlipCenteredKernelAlignment,
    'logCKA' : LogCenteredKernelAlignment,
    'fliplogCKA' : FlipLogCenteredKernelAlignment,
    'logCKA0' : LogCenteredKernelAlignment0
}
