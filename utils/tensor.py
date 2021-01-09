# -*- coding: utf-8 -*-
import torch

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
