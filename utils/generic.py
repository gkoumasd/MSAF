import torch
import random
import os
import numpy as np

def set_seed(params):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(params.seed)
    os.environ['PYTHONHASHSEED'] = str(params.seed)
    np.random.seed(params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(params.seed)
        torch.cuda.manual_seed_all(params.seed)
    else:
        torch.manual_seed(params.seed)
        
        
def to_one_hot(input_array):
    output = np.zeros((input_array.size, input_array.max()+1))
    output[np.arange(input_array.size),input_array] = 1
    return output

def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]

# Remove spurious values (nan, inf) of a given tensor
def clean_tensor(x):
    if not x.dtype == torch.float32:
        float_x = x.type(torch.float32)
        float_x[torch.isinf(float_x)] = 0
        float_x[torch.isnan(float_x)] = 0
        x = float_x.type(x.dtype)
    else:    
        x[torch.isinf(x)] = 0
        x[torch.isnan(x)] = 0
