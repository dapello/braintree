import torch as ch

class Adversary:
    """
    Assumes model takes inputs between [0-1], and X is normalized between [0-1].
    Uses PGD with Linf constraint.
    
    TODO: update to take model output at a specified layer.
    """
    def __init__(self, model, norm='inf', eps=1/255, 
             num_steps=5, eps_step_size=1/4, clamp=(0,1), batch_size=64):
        
        assert norm=='inf'
            
        self.model = model
        self.norm = norm
        self.eps = eps
        self.num_steps = num_steps
        self.eps_step_size = eps_step_size
        self.step_size = eps_step_size*eps
        self.clamp = clamp
        self.batch_size = batch_size
        
    def generate(self, X, Y, loss_fnc, output_inds=[0,1000]):
        """generate stimuli maximizing loss function provided"""
        training = self.model.training
        self.model.eval()
        X = X.cuda()
        Y = Y.cuda()
         
        with ch.set_grad_enabled(True):
            X_adv = []
            for X_, Y_ in zip(X.split(self.batch_size), Y.split(self.batch_size)):
                X_adv.append(self.generate_(X_, Y_, loss_fnc, output_inds=output_inds))
            X_adv = ch.cat(X_adv)
            
        if training:
            self.model.train()

        return X_adv

    def generate_(self, X, Y, loss_fnc, output_inds=[0,1000]):
        """generate stimuli maximizing loss function provided"""
        self.model.eval()
         
        with ch.set_grad_enabled(True):
            X_adv = X.clone().detach().requires_grad_(True).to(X.device)
            
            for i in range(self.num_steps):
                _X_adv = X_adv.clone().detach().requires_grad_(True).to(X.device)
                
                # calculate gradients w.r.t. loss
                loss_fnc(self.model(_X_adv)[:, output_inds[0]:output_inds[1]], Y).backward()

                # step in direction to increase loss
                X_adv = X_adv + self.step_size * _X_adv.grad.sign()
            
                # project onto eps sized ball, don't go outside (0,1) image bounds
                X_adv = ch.max(ch.min(X_adv, X + self.eps), X - self.eps).clamp(*self.clamp)
            
        return X_adv

    #def generate(self, X, Y, loss_fnc, output_inds=[0,1000]):
    #    """generate stimuli maximizing loss function provided"""
    #    self.model.eval()
    #     
    #    with ch.set_grad_enabled(True):
    #        X_adv = X.clone().detach().requires_grad_(True).to(X.device)
    #        
    #        for i in range(self.num_steps):
    #            _X_adv = X_adv.clone().detach().requires_grad_(True).to(X.device)
    #            
    #            # calculate gradients w.r.t. loss
    #            loss_fnc(self.model(_X_adv)[:, output_inds[0]:output_inds[1]], Y).backward()

    #            # step in direction to increase loss
    #            X_adv = X_adv + self.step_size * _X_adv.grad.sign()
    #        
    #            # project onto eps sized ball, don't go outside (0,1) image bounds
    #            X_adv = ch.max(ch.min(X_adv, X + self.eps), X - self.eps).clamp(*self.clamp)
    #        
    #    return X_adv
