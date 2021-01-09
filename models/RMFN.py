# -*- coding: utf-8 -*-

#CMU Multimodal SDK, CMU Multimodal Model SDK

#Multimodal Language Analysis with Recurrent Multistage Fusion, Paul Pu Liang, Ziyin Liu, Amir Zadeh, Louis-Philippe Morency - https://arxiv.org/abs/1808.03920 

#in_dimensions: the list of dimensionalities of each modality 

#cell_size: lstm cell size

#in_modalities: is a list of inputs from each modality - the first dimension of all the modality inputs must be the same, it will be the batch size. The second dimension is the feature dimension. There are a total of n modalities.

#steps: number of iterations for the recurrent fusion

import torch
from torch import nn
import torch.nn.functional as F
from .LSTHM import LSTHMCell
from .SimpleNet import SimpleNet

class RMFN(nn.Module):
    def __init__(self,opt):
        super(RMFN,self).__init__()
        self.input_dims = opt.input_dims
        self.output_dim = opt.output_dim
        if type(opt.hidden_dims) == int:
            self.hidden_dims = [opt.hidden_dims]
        else:
            self.hidden_dims = [int(s) for s in opt.hidden_dims.split(',')]
        self.num_modalities = len(self.input_dims)

        self.steps = opt.steps
        self.device = opt.device
        self.compression_cell_dim = opt.compression_cell_dim
        self.compression_dropout_rate = opt.compression_dropout_rate
        self.hlt_memory_init_cell_dim = opt.hlt_memory_init_cell_dim
        self.hlt_memory_init_dropout_rate = opt.hlt_memory_init_dropout_rate
        self.output_cell_dim = opt.output_cell_dim
        self.output_dropout_rate = opt.output_dropout_rate
   
        # initialize the LSTHM for each modality
        self.compressed_dim = opt.compressed_dim         
        if opt.embedding_enabled:
            embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
            self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=not opt.embedding_trainable)
        
        self.lsthms = nn.ModuleList([LSTHMCell(hidden_dim,input_dim,self.compressed_dim,self.device) \
                                     for input_dim,hidden_dim in zip(self.input_dims,self.hidden_dims)])
        
        self.hlt_memory_init_net = SimpleNet(sum(self.hidden_dims), self.hlt_memory_init_cell_dim,
                                             self.hlt_memory_init_dropout_rate, sum(self.hidden_dims),nn.Sigmoid())
        
        
        self.compression_net = SimpleNet(self.steps*sum(self.hidden_dims), self.compression_cell_dim,
                                         self.compression_dropout_rate,self.compressed_dim,nn.Sigmoid())

        
        self.recurrent_fusion = RecurrentFusion(self.hidden_dims, self.hlt_memory_init_net,self.compression_net,self.steps,self.device)
        
        if self.output_dim == 1:
            self.fc_out = SimpleNet(self.compressed_dim+sum(self.hidden_dims),self.output_cell_dim,
                                    self.output_dropout_rate,self.output_dim)
        else:
            self.fc_out = SimpleNet(self.compressed_dim+sum(self.hidden_dims),self.output_cell_dim,
                                    self.output_dropout_rate,self.output_dim, nn.Softmax(dim = 1))
#        self.fc_out = nn.Sequential(nn.Linear(self.compressed_dim+sum(self.hidden_dims),self.output_cell_dim),
#                                    nn.ReLU(),
#                                    nn.Dropout(self.output_dropout_rate),
#                                    nn.Linear(self.output_cell_dim,self.output_dim)
#                                    )
        
    def forward(self,in_modalities):
        in_modalities = [self.embed(modality) if len(modality.shape) == 2 \
                       else modality for modality in in_modalities]
        self.batch_size = in_modalities[0].shape[0]
        time_stamps = in_modalities[0].shape[1]
        z = torch.zeros(self.batch_size,self.compressed_dim).to(self.device)
        h = [torch.zeros(self.batch_size,hidden_dim).to(self.device) for hidden_dim in self.hidden_dims]
        c = [torch.zeros(self.batch_size,hidden_dim).to(self.device) for hidden_dim in self.hidden_dims]
        for t in range(time_stamps):
            for i in range(self.num_modalities):
                in_modality = in_modalities[i][:,t,:]
                c[i],h[i] = self.lsthms[i](in_modality,c[i],h[i],z)
                
            z = self.recurrent_fusion(h)
            z = torch.softmax(z,dim = 1)
        
        # Concatenate z with h
        total_output = torch.cat([z,*h],dim = 1)
        output = self.fc_out(total_output)
        return(output)
        

class RecurrentFusion(nn.Module):
    
    def __init__(self,in_dimensions,memory_init_net, compression_net, steps, device = torch.device('cpu')):
        super(RecurrentFusion, self).__init__()
        self.in_dimensions=in_dimensions
        self.total_in_size = sum(in_dimensions)
        self.steps = steps
        self.device = device
#        self.compression_cell_dim = compression_cell_dim
#        self.model=nn.LSTM(self.total_in_size,cell_size)
        self.hlt_memory_init = memory_init_net
        
        self.hlt_W=nn.Linear(self.total_in_size,4*self.total_in_size).to(self.device)
        self.hlt_U=nn.Linear(self.total_in_size, 4*self.total_in_size).to(self.device)
        
        
        self.fuse_W = nn.Linear(self.total_in_size, 4*self.total_in_size).to(self.device)
        self.fuse_U = nn.Linear(self.total_in_size, 4*self.total_in_size).to(self.device)
        self.compression_net = compression_net

    def forward(self,in_modalities):
        batch_size=in_modalities[0].shape[0]
        
        # Highlight LSTM
#        model_input = torch.cat(in_modalities,dim=1)
        model_input = torch.zeros(batch_size, self.total_in_size).to(self.device)
        c_hlt = self.hlt_memory_init(model_input)
        h_hlt = torch.zeros(batch_size,self.total_in_size).to(self.device)

        inp = model_input
        output_hlt = []
        for i in range(self.steps):
            input_affine = self.hlt_W(inp)
            output_affine = self.hlt_U(h_hlt)   
            sums = input_affine + output_affine
            
            f_t=torch.sigmoid(sums[:,:self.total_in_size])
            i_t=torch.sigmoid(sums[:,self.total_in_size:2*self.total_in_size])
            o_t=torch.sigmoid(sums[:,2*self.total_in_size:3*self.total_in_size])
            ch_t=torch.tanh(sums[:,3*self.total_in_size:])
            c_hlt=f_t*c_hlt+i_t*ch_t
            h_hlt=torch.tanh(c_hlt)*o_t
            
            # Attention
            h_hlt = torch.softmax(h_hlt,dim = 1)
            inp = model_input*h_hlt
            output_hlt.append(inp)
            
        # FUSE LSTM
        # random orthogonal initialization, to be implemented later
        c_fuse = torch.rand(batch_size, self.total_in_size).to(self.device)
        h_fuse = torch.zeros(batch_size, self.total_in_size).to(self.device)
        output_fuse = []
        for i in range(self.steps):
            input_affine=self.fuse_W(output_hlt[i])
            output_affine=self.fuse_U(h_fuse)   
            sums=input_affine+output_affine
            
            f_t=torch.sigmoid(sums[:,:self.total_in_size])
            i_t=torch.sigmoid(sums[:,self.total_in_size:2*self.total_in_size])
            o_t=torch.sigmoid(sums[:,2*self.total_in_size:3*self.total_in_size])
            ch_t=torch.tanh(sums[:,3*self.total_in_size:])
            c_fuse=f_t*c_fuse+i_t*ch_t
            h_fuse=torch.tanh(c_fuse)*o_t
            output_fuse.append(h_fuse)
        
        z = self.compression_net(torch.cat(output_fuse,dim =1))
        return z

if __name__=="__main__":
    print("This is a module and hence cannot be called directly ...")
    print("A toy sample will now run ...")
    
    from torch.autograd import Variable
    import torch.nn.functional as F
    import numpy

    inputx=Variable(torch.Tensor(numpy.zeros([32,40])),requires_grad=True)
    inputy=Variable(torch.Tensor(numpy.array(numpy.zeros([32,12]))),requires_grad=True)
    inputz=Variable(torch.Tensor(numpy.array(numpy.zeros([32,20]))),requires_grad=True)
    modalities=[inputx,inputy,inputz]
    
    fmodel=RecurrentFusion([40,12,20],100)
    
    out=fmodel(modalities,steps=5)

    print("Output")
    print(out[0])
    print("Toy sample finished ...")


