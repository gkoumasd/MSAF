# -*- coding: utf-8 -*-

import torch
from torch import nn
from .SimpleNet import SimpleNet

class MFN(nn.Module):
    def __init__(self,opt):
        super(MFN, self).__init__()
        
        self.device = opt.device
        
        self.input_dims = opt.input_dims
        self.num_modalities = len(self.input_dims)      
        
        if type(opt.hidden_dims) == int:
            self.hidden_dims = [opt.hidden_dims]
        else:
            self.hidden_dims = [int(s) for s in opt.hidden_dims.split(',')]
            
        self.total_cell_dim = sum(self.hidden_dims)
        self.memory_dim = opt.memory_dim

        self.window_dim = opt.window_dim
        self.output_dim = opt.output_dim
        
        self.attn_in_dim = self.total_cell_dim * self.window_dim 
        self.gamma_in_dim = self.attn_in_dim + self.memory_dim        
        self.final_fc_in_dim = self.total_cell_dim + self.memory_dim
        
        
        if type(opt.attn_cell_dims) == int:
            self.attn_cell_dims = [opt.attn_cell_dims]
        else:
            self.attn_cell_dims = [int(s) for s in opt.attn_cell_dims.split(',')]   
        
        if type(opt.gamma_cell_dims) == int:
            self.gamma_cell_dims = [opt.gamma_cell_dims]
        else:
            self.gamma_cell_dims = [int(s) for s in opt.gamma_cell_dims.split(',')]   
            
        self.output_cell_dim = opt.output_cell_dim
        
        if type(opt.attn_dropout_rates) == float:
            self.attn_dropout_rates = [opt.attn_dropout_rates]
        else:
            self.attn_dropout_rates = [float(s) for s in opt.attn_dropout_rates.split(',')]   
        
        if type(opt.gamma_dropout_rates) == float:
            self.gamma_dropout_rates = [opt.gamma_dropout_rates]
        else:
            self.gamma_dropout_rates = [float(s) for s in opt.gamma_dropout_rates.split(',')]  
         
        self.out_dropout_rate = opt.out_dropout_rate
        if opt.embedding_enabled:
            embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
            self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=not opt.embedding_trainable)
        self.lstms = nn.ModuleList([nn.LSTMCell(input_dim, cell_dim) \
                                    for input_dim, cell_dim in zip(self.input_dims, self.hidden_dims)])
        
         
        self.fc_attn_1 = SimpleNet(self.attn_in_dim, self.attn_cell_dims[0],
                                   self.attn_dropout_rates[0],self.attn_in_dim,
                                   nn.Softmax(dim = 1)) 

        self.fc_attn_2 = SimpleNet(self.attn_in_dim, self.attn_cell_dims[1],
                                   self.attn_dropout_rates[1],self.memory_dim,
                                   nn.Tanh())
        
        self.fc_gamma = nn.ModuleList([SimpleNet(self.gamma_in_dim, cell_dim,
                                   dropout,self.memory_dim,
                                   nn.Sigmoid()) for cell_dim, dropout in zip(self.gamma_cell_dims,self.gamma_dropout_rates)])
        
        if self.output_dim == 1:
            self.fc_out = SimpleNet(self.final_fc_in_dim, self.output_cell_dim,
                                    self.out_dropout_rate,self.output_dim)
            
        else:
            self.fc_out = SimpleNet(self.final_fc_in_dim, self.output_cell_dim,
                                    self.out_dropout_rate,self.output_dim, nn.Softmax(dim = 1))

    def forward(self,in_modalities):
        in_modalities = [self.embed(modality) if len(modality.shape) == 2 \
                       else modality for modality in in_modalities]
        self.batch_size = in_modalities[0].shape[0]
        time_stamps = in_modalities[0].shape[1]

        self.h = [torch.zeros(self.batch_size, cell_dim).to(self.device) for cell_dim in self.hidden_dims]
        self.c = [torch.zeros(self.batch_size, cell_dim).to(self.device) for cell_dim in self.hidden_dims]      
        self.memory = torch.zeros(self.batch_size, self.memory_dim).to(self.device)        
        all_h = []  
        all_c = []      
        all_memory = []        
        for t in range(time_stamps):
            # prev time step
            prev_c = self.c
            new_h = []
            new_c = []
            for i in range(self.num_modalities):
                modality = in_modalities[i][:,t,:]
                h_m, c_m = self.lstms[i](modality, (self.h[i],self.c[i]))
                new_h.append(h_m)
                new_c.append(c_m)
                      
            # concatenate
            prev_c_cat = torch.cat(prev_c, dim = 1)
            new_c_cat = torch.cat(new_c,dim = 1)
            
            
            c_star = torch.cat([prev_c_cat, new_c_cat],dim = 1)          
            attention = self.fc_attn_1(c_star)      
            c_star_attn = attention * c_star
            c_hat = self.fc_attn_2(c_star_attn)
            
            
            temp_memory = torch.cat([c_star_attn, self.memory], dim = 1)         
            gamma1 = self.fc_gamma[0](temp_memory)         
            gamma2 = self.fc_gamma[1](temp_memory) 

            self.memory = gamma1*self.memory + gamma2*c_hat
            all_memory.append(self.memory)
            
            # updates
            self.h = new_h
            self.c = new_c
            
            all_h.append(self.h)
            all_c.append(self.c)

        # last hidden layer last_hs is n x h
        last_h = all_h[-1]
        last_memory = all_memory[-1]  
        total_output = torch.cat([*last_h,last_memory], dim=1)   
        output = self.fc_out(total_output)
        return output
