# -*- coding: utf-8 -*-
import torch
import numpy as np

def get_remaining_parameters(model, parameter_list):
    all_params = list(model.parameters())
    params_list = []
    for ind, p in enumerate(all_params):
        isin = False
        for p_l in parameter_list:
            if torch.equal(p,p_l):
                isin = True
                break
        if not isin:
            params_list.append(p)
            
    return params_list

def flatten_output(targets, outputs):
    new_targets = []
    new_outputs = []
    for i in range(len(targets)):
        target = targets[i]
        for j in range(len(target)):
            t = target[j]
            if len(t) > 0 and not torch.sum(t) == 0:
                new_targets.append(t)
                new_outputs.append(outputs[i][j])
    new_targets = torch.stack(new_targets)
    new_outputs = torch.stack(new_outputs)
    return new_targets, new_outputs


def get_unitary_parameters(model):
    params_list = []
    if 'recurrent_cells' in model._modules:
        
        rnn_cells = model.recurrent_cells
        for cell in rnn_cells:
            params_list.append(cell.unitary_x)
            params_list.append(cell.unitary_h)
    
    if 'dense' in model._modules:
        params_list.append(model.dense.weight)
    if 'measurement' in model._modules:
        params_list.append(model.measurement.kernel)
    
    return params_list