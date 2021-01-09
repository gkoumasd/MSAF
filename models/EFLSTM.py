# -*- coding: utf-8 -*-
import torch
from torch import nn
from .SimpleNet import SimpleNet

class EFLSTM(nn.Module):
    def __init__(self, opt):
        super(EFLSTM, self).__init__()
        self.device = opt.device    
        self.input_dims = opt.input_dims
        self.total_input_dim = sum(self.input_dims)
        self.hidden_dim = opt.hidden_dim
        if opt.embedding_enabled:
            embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
            self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze= not opt.embedding_trainable)
            
        if self.hidden_dim == -1:
            self.hidden_dim = self.total_input_dim
        self.output_dim = opt.output_dim
        self.output_cell_dim = opt.output_cell_dim
        self.output_dropout_rate = opt.output_dropout_rate
        self.lstm = nn.LSTMCell(self.total_input_dim, self.hidden_dim)
        
        self.fc_out = SimpleNet(self.hidden_dim,self.output_cell_dim,
                                self.output_dropout_rate,self.output_dim)


    def forward(self, in_modalities):
        in_modalities = [self.embed(modality) if len(modality.shape) == 2 \
                       else modality for modality in in_modalities]
#        print(in_modalities[0].shape)
#        print(in_modalities[1].shape)
#        print(in_modalities[2].shape)
        batch_input = torch.cat(in_modalities,dim =2)
        time_stamps = batch_input.shape[1]
        batch_size = batch_input.shape[0]
        self.h = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        self.c = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        
        all_h = []
        all_c = []
        for t in range(time_stamps):
            self.h, self.c = self.lstm(batch_input[:,t,:], (self.h,self.c))
            all_h.append(self.h)
            all_c.append(self.c)
             
        # last hidden layer last_hs is n x h
        last_h = all_h[-1]
        output = self.fc_out(last_h)
        return output