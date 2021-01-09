# -*- coding: utf-8 -*-

#CMU Multimodal SDK, CMU Multimodal Model SDK

#Multi-attention Recurrent Network for Human Communication Comprehension, Amir Zadeh, Paul Pu Liang, Soujanya Poria, Erik Cambria, Prateek Vij, Louis-Philippe Morency - https://arxiv.org/pdf/1802.00923.pdf

#in_modalities: is a list of inputs from each modality - the first dimension of all the modality inputs must be the same, it will be the batch size. The second dimension is the feature dimension. There are a total of n modalities.

#attention_model: is a pytorch nn.Sequential which takes in an input with size (bs * m0+...+mn) with m_i being the dimensionality of the features in modality i. Output is the (bs * (m0+...+mn)*num_atts).

#dim_reduce_nets: is a list of pytorch nn.Sequential which takes in an input with size (bs*(mi*num_atts))
#num_atts is the number of attentions

#num_atts: number of attentions


import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
from .LSTHM import LSTHMCell
from .SimpleNet import SimpleNet

class MARN(nn.Module):
    def __init__(self,opt):
        super(MARN,self).__init__()
        self.input_dims = opt.input_dims
        self.output_dim = opt.output_dim
        if type(opt.hidden_dims) == int:
            self.hidden_dims = [opt.hidden_dims]
        else:
            self.hidden_dims = [int(s) for s in opt.hidden_dims.split(',')]
        self.device = opt.device
        self.attn_num = opt.attn_num
        self.attn_cell_dim = opt.attn_cell_dim
        self.attn_dropout_rate = opt.attn_dropout_rate
        if opt.embedding_enabled:
            embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
            self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=not opt.embedding_trainable)
            
        if type(opt.compression_cell_dims) == int:
            self.compression_cell_dims = [opt.compression_cell_dims]
        else:
            self.compression_cell_dims = [int(s) for s in opt.compression_cell_dims.split(',')]
        
        if type(opt.compression_dropout_rates) == float:
            self.compression_dropout_rates = [opt.compression_dropout_rates]
        else:
            self.compression_dropout_rates = [float(s) for s in opt.compression_dropout_rates.split(',')]
        self.total_hidden_dim = sum(self.hidden_dims)
        self.output_cell_dim = opt.output_cell_dim
        self.output_dropout_rate = opt.output_dropout_rate
                    
#        self.batch_size = opt.batch_size
        
        # initialize the LSTHM for each modalitye
        
        if type(opt.compressed_dims) == int:
            self.compressed_dims = [opt.compressed_dims]
        else:
            self.compressed_dims = [int(s) for s in opt.compressed_dims.split(',')]
        self.num_modalities = len(self.input_dims)
        self.lsthms = nn.ModuleList([LSTHMCell(hidden_dim,input_dim,sum(self.compressed_dims),self.device) \
                                     for input_dim,hidden_dim in zip(self.input_dims,self.hidden_dims)])
        
        attention = SimpleNet(self.total_hidden_dim, self.attn_cell_dim*self.attn_num,
                              self.attn_dropout_rate,self.total_hidden_dim*self.attn_num,nn.Sigmoid())
             
        compression_nets = nn.ModuleList([SimpleNet(hidden_dim*self.attn_num, compression_cell_dim,
                                                    compression_dropout_rate,compressed_dim,nn.Sigmoid())
                                            for compressed_dim,hidden_dim, compression_cell_dim, compression_dropout_rate
                                    in zip(self.compressed_dims,self.hidden_dims,self.compression_cell_dims,self.compression_dropout_rates)])
        
        self.multi_attention = MultipleAttention(attention, compression_nets, self.attn_num)
        self.fc_output_in_dim = sum(self.compressed_dims)+self.total_hidden_dim
        
       
        self.fc_out = SimpleNet(self.fc_output_in_dim, self.output_cell_dim,
                                self.output_dropout_rate, self.output_dim)
        
    
    def forward(self, in_modalities):
        in_modalities = [self.embed(modality) if len(modality.shape) == 2 \
                       else modality for modality in in_modalities]
        self.batch_size = in_modalities[0].shape[0]
        time_stamps = in_modalities[0].shape[1]
        h = [torch.zeros(self.batch_size,hidden_dim).to(self.device) for hidden_dim in self.hidden_dims]
        c = [torch.zeros(self.batch_size,hidden_dim).to(self.device) for hidden_dim in self.hidden_dims]
        z = torch.zeros(self.batch_size,sum(self.compressed_dims)).to(self.device)

        for t in range(time_stamps):
            for i in range(self.num_modalities):              
                in_modality = in_modalities[i][:,t,:]
                c[i],h[i] = self.lsthms[i](in_modality,c[i],h[i],z)
            output = self.multi_attention(h)
            z = torch.cat(output[0],dim = 1)
            z = torch.softmax(z, dim = 1)
        
        # Concatenate z with h
        total_output = torch.cat([z,*h],dim = 1)
        output = self.fc_out(total_output)

        return(output)
        
        
                
class MultipleAttention(nn.Module):

    def __init__(self,attention_model,dim_reduce_nets,num_atts):
        super(MultipleAttention, self).__init__()
        self.attention_model=attention_model
        self.dim_reduce_nets=dim_reduce_nets
        self.num_atts=num_atts

    def forward(self,in_modalities):

        batch_size = in_modalities[0].shape[0]
        
#        input_dim = sum([modality.shape[1] for modality in in_modalities])
        #getting some simple integers out
        num_modalities = len(in_modalities)
        
        #simply the tensor that goes into attention_model
        in_tensor = torch.cat(in_modalities,dim=1)    
        attention = self.attention_model(in_tensor).view(batch_size,self.num_atts,-1)
        
        #calculating attentions
        atts=torch.softmax(attention,dim=2).view(batch_size,-1)
        
        #calculating the tensor that will be multiplied with the attention
        out_tensor=torch.cat([in_modalities[i].repeat(1,self.num_atts) for i in range(num_modalities)],dim=1)
        
        #calculating the attention
        att_out=atts*out_tensor
        
        #now to apply the dim_reduce networks
        #first back to however modalities were in the problem
        start=0
        out_modalities=[]
        for i in range(num_modalities):
            modality_length=in_modalities[i].shape[1]*self.num_atts
            out_modalities.append(att_out[:,start:start+modality_length])
            start=start+modality_length
    
        #apply the dim_reduce
        dim_reduced=[self.dim_reduce_nets[i](out_modalities[i]) for i in range(num_modalities)]
        #multiple attention done :)
        
        return dim_reduced,out_modalities


if __name__=="__main__":
    print("This is a module and hence cannot be called directly ...")
    print("A toy sample will now run ...")
    
    from torch.autograd import Variable
    import torch.nn.functional as F
    import numpy

    inputx=Variable(torch.Tensor(numpy.array(numpy.zeros([32,40]))),requires_grad=True)
    inputy=Variable(torch.Tensor(numpy.array(numpy.zeros([32,12]))),requires_grad=True)
    inputz=Variable(torch.Tensor(numpy.array(numpy.zeros([32,20]))),requires_grad=True)
    modalities=[inputx,inputy,inputz]

    #simple functions for toy example, 4 times extract attentions hence 72*4
    my_attention = nn.Sequential(nn.Linear(72,72*4))
    small_netx = nn.Sequential(nn.Linear(160,10))
    small_nety = nn.Sequential(nn.Linear(48,20))
    small_netz = nn.Sequential(nn.Linear(80,30))

    smalls_nets=[small_netx,small_nety,small_netz]

    fmodel=MultipleAttention(my_attention,smalls_nets,4)    
    out=fmodel(modalities)

    print("Output")
    print(out[0])
    print("Toy sample finished ...")
