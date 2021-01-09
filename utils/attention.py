import torch
from torch import nn
import numpy as np

def bi_modal_attention(x, y):
        
    ''' 
    .  stands for dot product 
    *  stands for elemwise multiplication
    {} stands for concatenation
    
    m1 = x . transpose(y) ||  m2 = y . transpose(x) 
    n1 = softmax(m1)      ||  n2 = softmax(m2)
    o1 = n1 . y           ||  o2 = m2 . x
    a1 = o1 * x           ||  a2 = o2 * y
    
    return {a1, a2}

    '''
    softmax = nn.Softmax(dim = -1)
    
    m1 = torch.mm(x, y.permute(1,0))
    n1 = softmax(m1)
    o1 = torch.mm(n1, y) 
    a1 = o1 * x
    
    m2 = torch.mm(y, x.permute(1,0))
    n2 = softmax(m2)
    o2 = torch.mm(n2, x)
    a2 = o2 * y
        
        
    return torch.cat([a1, a2], dim=-1)



def uni_modal_attention(x, y):
        
    ''' 
    .  stands for dot product 
    *  stands for elemwise multiplication
    {} stands for concatenation
    
    m1 = x . transpose(y) 
    n1 = softmax(m1)      
    o1 = n1 . y           
    a1 = o1 * x
    
    return {a1, a2}

    '''
    softmax = nn.Softmax(dim = -1)
    
    m = torch.bmm(x, y.permute(0,2,1))
    n = softmax(m)
    o = torch.bmm(n, y) 
    a = o * x
    
    return a

    
    
def self_attention(x):
    
    ''' 
    .  stands for dot product 
    *  stands for elemwise multiplication
    
    m = x . transpose(x)
    n = softmax(m)
    o = n . x  
    a = o * x           
    
    return a

    '''
    softmax = nn.Softmax(dim = -1)
    
    m = torch.bmm(x, x.permute(0,2,1))
    n = softmax(m)
    o = torch.bmm(n, x)
    a = o * x
    
    return a   



