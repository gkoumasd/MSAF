# -*- coding: utf-8 -*-
import torch

def step_unitary(G_r, G_i, W_r, W_i, lr):

    # A = G^H W - W^H G
    # A_r = (G_r^T W_r)+ (G_i^T W_i)- (W_r^T G_r) - (W_i^T G_i)
    # A_i = (G_r^T W_i)- (G_i^T W_r)- (W_r^T G_i)+ (W_i^T G_r)
                
    A_skew_r = torch.mm(G_r.t(),W_r) + torch.mm(G_i.t(),W_i) - torch.mm(W_r.t(),G_r) -  torch.mm(W_i.t(),G_i)
    A_skew_i = torch.mm(G_r.t(),W_i) - torch.mm(G_i.t(),W_r) - torch.mm(W_r.t(),G_i) +  torch.mm(W_i.t(),G_r)
    
    #W_new = (I+lr/2 * A)^(-1)*(I-lr/2 * A)*W
    idm = torch.eye(G_r.shape[0]).to(G_r.device)

    
    # cayley_numer = I-lr/2 * A
    cayley_numer_r = idm + (lr/2)* A_skew_r
    cayley_numer_i = + (lr/2)* A_skew_i
    
    # cayley_demon = (I + lr/2 * A)^(-1)
    X = idm - (lr/2)* A_skew_r
    Y = - (lr/2)* A_skew_i
    
    #(X + i*Y)^-1 = (X + Y*X^-1*Y)^-1 - i*(Y + X*Y^-1*X)^-1
    
    #cayley_denom_r = (X + torch.mm(Y,torch.mm(X.inverse(),Y))).inverse()
    
    if X.det() == 0:
        X.add_(idm,alpha=1e-5)
    
    if Y.det() == 0:
        Y.add_(idm,alpha=1e-5)
    
    inv_cayley_denom_r = X + torch.mm(Y,torch.mm(X.inverse(),Y))
    if inv_cayley_denom_r.det() == 0:
        inv_cayley_denom_r.add_(idm,alpha=1e-5)
    
    cayley_denom_r = inv_cayley_denom_r.inverse()
    
    #cayley_denom_i = - (Y + torch.mm(X,torch.mm(Y.inverse(),X))).inverse()
    inv_cayley_denom_i = Y + torch.mm(X,torch.mm(Y.inverse(),X))
    if inv_cayley_denom_i.det() == 0:
        inv_cayley_denom_i.add_(idm,alpha=1e-5)
    
    cayley_denom_i = - inv_cayley_denom_i.inverse()
    
    #W_new = cayley_denom*cayley_numer*W
    W_new_r = torch.mm(cayley_denom_r, cayley_numer_r) - torch.mm(cayley_denom_i, cayley_numer_i)
    W_new_i = torch.mm(cayley_denom_r, cayley_numer_i) + torch.mm(cayley_denom_i, cayley_numer_r)            
    
    W_new_r_2 = torch.mm(W_new_r, W_r) - torch.mm(W_new_i, W_i)
    W_new_i_2 = torch.mm(W_new_r, W_i) + torch.mm(W_new_i, W_r)
    
    return torch.stack([W_new_r_2, W_new_i_2], dim= -1)