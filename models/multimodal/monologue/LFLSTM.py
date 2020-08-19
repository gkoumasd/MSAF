# -*- coding: utf-8 -*-
import torch
from torch import nn
from .SimpleNet import SimpleNet

class LFLSTM(nn.Module):
    def __init__(self, opt): #, n_layers, bidirectional, dropout):
        super(LFLSTM, self).__init__()
        self.device = opt.device
        self.input_dims = opt.input_dims
        if type(opt.hidden_dims) == int:
            self.hidden_dims = [opt.hidden_dims]
        else:
            self.hidden_dims = [int(s) for s in opt.hidden_dims.split(',')]
             
        if opt.embedding_enabled:
            embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
            self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
            
        self.num_modalities = len(self.input_dims)
        self.output_dim = opt.output_dim
        self.output_cell_dim = opt.output_cell_dim
        self.output_dropout_rate = opt.output_dropout_rate
        
        self.lstms = nn.ModuleList([nn.LSTMCell(input_dim, hidden_dim) \
                                   for input_dim, hidden_dim in \
                                   zip(self.input_dims, self.hidden_dims)])
        
        if self.output_dim == 1:     
            self.fc_out = SimpleNet(sum(self.hidden_dims),self.output_cell_dim,
                                    self.output_dropout_rate,self.output_dim)
        else:
            self.fc_out = SimpleNet(sum(self.hidden_dims),self.output_cell_dim,
                                    self.output_dropout_rate,self.output_dim)


    def forward(self, in_modalities):
        in_modalities = [self.embed(modality) if len(modality.shape) == 2 \
                       else modality for modality in in_modalities]
        batch_size = in_modalities[0].shape[0]
        time_stamps = in_modalities[0].shape[1]
        all_h = []
        all_c = []
        for i in range(self.num_modalities):
            h = []
            c = []
            _h = torch.zeros(batch_size, self.hidden_dims[i]).to(self.device)
            _c = torch.zeros(batch_size, self.hidden_dims[i]).to(self.device)
            
            
            for t in range(time_stamps):
                _h, _c = self.lstms[i](in_modalities[i][:,t,:], (_h,_c))
                h.append(_h)
                c.append(_c)
            all_h.append(h)
            all_c.append(c)
            
        
        # last hidden layer last_hs is n x h
        last_h = torch.cat([hh[-1] for hh in all_h], dim = 1)
        output = self.fc_out(last_h)
        return output