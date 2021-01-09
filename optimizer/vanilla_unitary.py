# -*- coding: utf-8 -*-
import torch
from torch.optim.optimizer import Optimizer, required
from .utils import step_unitary

class Vanilla_Unitary(Optimizer):
    """Implements gradient descent for unitary matrix.
        
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        


    .. note::
        This is the vanilla version of the gradient descent for unitary matrix, 
        i.e. formula (6) in H. D. Tagare. Notes on optimization on Stiefel manifolds. 
        Technical report, Yale University, 2011, and formula (6) in Scott Wisdom, 
        Thomas Powers, John Hershey, Jonathan Le Roux, and Les Atlas. Full-capacity 
        unitary recurrentneural networks. In NIPS 2016. 

        .. math::
                  A = G^H*W - W^H*G \\
                  W_new = (I+lr/2 * A)^(-1)*(I-lr/2 * A)*W

        where W, G and lr denote the parameters, gradient
        and learning rate respectively.
    """

    def __init__(self, params, lr=required, device = torch.device('cpu')):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        self.device = device
        defaults = dict(lr=lr)
        super(Vanilla_Unitary, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Vanilla_Unitary, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
#            weight_decay = group['weight_decay']
#            momentum = group['momentum']
#            dampening = group['dampening']
#            nesterov = group['nesterov']
            lr = group['lr']
            
            for p in group['params']:
    
                if p.grad is None:
                    continue
                
                d_p = p.grad.data #G
                
                G_r = d_p[:,:,0]
                G_i = d_p[:,:,1]
                W_r = p.data[:,:,0]
                W_i = p.data[:,:,1]

                p.data = step_unitary(G_r, G_i, W_r, W_i, lr)


        return loss