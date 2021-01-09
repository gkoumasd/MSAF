# -*- coding: utf-8 -*-

import torch
from torch import nn
from .SimpleNet import SimpleNet
class LSTHMCell(nn.Module):

    def __init__(self,cell_size,in_size,hybrid_in_size,device = torch.device('cpu') ):
        super(LSTHMCell, self).__init__()
        self.cell_size=cell_size
        self.in_size=in_size
        self.device = device
        self.hybrid_in_size = hybrid_in_size
        self.W=nn.Linear(in_size,4*self.cell_size).to(self.device)
        self.U=nn.Linear(cell_size,4*self.cell_size).to(self.device)
        self.V=nn.Linear(hybrid_in_size,4*self.cell_size).to(self.device)
    
    
    def forward(self, input_x,input_c,input_h,input_z):
        input_affine=self.W(input_x).to(self.device)
        output_affine=self.U(input_h).to(self.device)
        hybrid_affine=self.V(input_z).to(self.device)
        
        sums=input_affine+output_affine+hybrid_affine

        #biases are already part of W and U and V
        f_t=torch.sigmoid(sums[:,:self.cell_size])
        i_t=torch.sigmoid(sums[:,self.cell_size:2*self.cell_size])
        o_t=torch.sigmoid(sums[:,2*self.cell_size:3*self.cell_size])
        ch_t=torch.tanh(sums[:,3*self.cell_size:])
        c_t=f_t*input_c+i_t*ch_t
        h_t=torch.tanh(c_t)*o_t
        return c_t,h_t
    
class LSTHM(nn.Module):
    
    def __init__(self,opt):
        super(LSTHM, self).__init__()
        
        self.input_dims = opt.input_dims
        self.output_dim = opt.output_dim
        
        if type(opt.hidden_dims) == int:
            self.hidden_dims = [opt.hidden_dims]
        else:
            self.hidden_dims = [int(s) for s in opt.hidden_dims.split(',')]
#        self.cell_size = opt.cell_size
        if opt.embedding_enabled:
            embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
            self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.num_modalities = len(self.input_dims)
        self.device = opt.device
        self.hybrid_in_size = opt.hybrid_in_size
        self.hybrid_cell_size = opt.hybrid_cell_size
        self.hybrid_dropout_rate = opt.hybrid_dropout_rate
        self.output_cell_dim = opt.output_cell_dim
        self.output_dropout_rate = opt.output_dropout_rate

        self.lsthms = nn.ModuleList([LSTHMCell(hidden_dim, input_dim,self.hybrid_in_size, \
                                              device = self.device) for hidden_dim, input_dim \
                                                in zip(self.hidden_dims, self.input_dims)])
    
        self.fc_hybrid = SimpleNet(sum(self.hidden_dims),self.hybrid_cell_size, 
                                   self.hybrid_dropout_rate,self.hybrid_in_size, nn.Sigmoid())
        
        if self.output_dim == 1:
            self.fc_out =  SimpleNet(sum(self.hidden_dims)+self.hybrid_in_size,self.output_cell_dim, 
                                   self.output_dropout_rate,self.output_dim)
        else:
            self.fc_out =  SimpleNet(sum(self.hidden_dims)+self.hybrid_in_size,self.output_cell_dim, 
                                   self.output_dropout_rate,self.output_dim,nn.Softmax(dim = 1))
            
        
    def forward(self, in_modalities):
        in_modalities = [self.embed(modality) if len(modality.shape) == 2 \
                       else modality for modality in in_modalities]
        self.batch_size = in_modalities[0].shape[0]
        time_stamps = in_modalities[0].shape[1]
        h = [torch.zeros(self.batch_size,hidden_dim).to(self.device) for hidden_dim in self.hidden_dims]
        c = [torch.zeros(self.batch_size,hidden_dim).to(self.device) for hidden_dim in self.hidden_dims]
        z = torch.zeros(self.batch_size, self.hybrid_in_size).to(self.device)
        all_h = []
        for t in range(time_stamps):
            for i in range(self.num_modalities):
                c[i],h[i] = self.lsthms[i](in_modalities[i][:,t,:], c[i],h[i],z)
            h_concat = torch.cat(h,dim = 1)
            z = self.fc_hybrid(h_concat)
            all_h.append(h)
        last_h = all_h[-1]
        total_output = torch.cat([*last_h,z], dim=1)   
        output = self.fc_out(total_output)         
        return output