import torch as ch

class Adversary:
    """
    Assumes model takes inputs between [0-1], and X is normalized between [0-1].
    Uses PGD with Linf constraint.
    
    TODO: update to take model output at a specified layer.
    """
    def __init__(self, model, norm='inf', eps=1/255, 
             num_steps=16, eps_step_size=1/4, clamp=(0,1)):
        
        assert norm=='inf'
            
        self.model = model
        self.norm = norm
        self.eps = eps
        self.num_steps = num_steps
        self.eps_step_size = eps_step_size
        self.step_size = eps_step_size*eps
        self.clamp = clamp
        
    def generate(self, X, Y, loss_fnc):
        """generate stimuli maximizing loss function provided"""
        self.model.eval()
            
        X_adv = X.clone().detach().requires_grad_(True).to(X.device)
        
        for i in range(self.num_steps):
            _X_adv = X_adv.clone().detach().requires_grad_(True).to(X.device)
            
            # calculate gradients w.r.t. loss
            loss_fnc(self.model(_X_adv), Y).backward()

            # step in direction to increase loss
            X_adv = X_adv + self.step_size * _X_adv.grad.sign()
        
            # project onto eps sized ball, don't go outside (0,1) image bounds
            X_adv = ch.max(ch.min(X_adv, X + self.eps), X - self.eps).clamp(*self.clamp)
            
        return X_adv
