# -*- coding: utf-8 -*-

#CMU Multimodal SDK, CMU Multimodal Model SDK

#Efficient Low-rank Multimodal Fusion with Modality-Specific Factors, Zhun Liu∗, Ying Shen∗, Varun Bharadhwaj Lakshminarasimhan,Paul Pu Liang, Amir Zadeh, Louis-Philippe Morency - https://arxiv.org/pdf/1806.00064.pdf

#in_modalities: is a list of inputs from each modality - the first dimension of all the modality inputs must be the same, it will be the batch size. The second dimension is the feature dimension. There are a total of n modalities.

#out_dimension: the output of the tensor fusion

import torch
import time
from torch import nn
import torch.nn.functional as F
from six.moves import reduce
from torch.autograd import Variable
import torch.nn.functional as F
import numpy
from torch.nn.init import xavier_normal_ as xavier_normal


class MLPSubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(MLPSubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        x = torch.mean(x, dim = 1,keepdim = False)
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = torch.relu(self.linear_1(dropped))
        y_2 = torch.relu(self.linear_2(y_1))
        y_3 = torch.relu(self.linear_3(y_2))

        return y_3


class LSTMSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in TFN for text
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(LSTMSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        if len(y_1.shape) == 1:
            y_1 = torch.unsqueeze(y_1,0)
        return y_1
    
class LMF(nn.Module):
    
    def __init__(self,opt):    
        super(LMF, self).__init__()
        
        self.input_dims = opt.input_dims
        if opt.embedding_enabled:
            embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
            self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=not opt.embedding_trainable)
        self.output_dim = opt.output_dim
        self.device = opt.device
        self.rank = opt.rank
        self.num_modalities = len(self.input_dims)
        self.use_softmax = opt.use_softmax
        
        if type(opt.hidden_dims) == int:
            self.hidden_dims = [opt.hidden_dims]
        else:
            self.hidden_dims = [int(s) for s in opt.hidden_dims.split(',')]
            
        self.text_out_dim = int(self.input_dims[0]/2)
        
        self.tensor_size = self.text_out_dim+1 
        for d in self.hidden_dims[1:]:
            self.tensor_size = self.tensor_size * (d+1) 
        
        if type(opt.dropout_probs) == float:
            self.dropout_probs = [opt.dropout_probs]
        else:
            self.dropout_probs = [float(s) for s in opt.dropout_probs.split(',')]
        

        # define the pre-fusion subnetworks

        self.text_subnet = LSTMSubNet(self.input_dims[0], self.hidden_dims[0], \
                                      self.text_out_dim, dropout = self.dropout_probs[0])
        
        self.other_subnets = nn.ModuleList([MLPSubNet(input_dim, hidden_dim, dropout_prob) \
                                            for input_dim, hidden_dim, dropout_prob in \
                                            zip(self.input_dims[1:],self.hidden_dims[1:],self.dropout_probs[1:])])
    
        # define the post_fusion layers
        self.text_factor = nn.Parameter(torch.Tensor(self.rank, self.text_out_dim + 1, self.output_dim).to(self.device))
        self.other_factors = [nn.Parameter(torch.Tensor(self.rank, hidden_dim + 1, self.output_dim).to(self.device)) for hidden_dim in self.hidden_dims[1:]]
                
        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank).to(self.device))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.output_dim).to(self.device))
        
        xavier_normal(self.text_factor)
        for factor in self.other_factors:
            xavier_normal(factor)
        xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)
    
    
    def forward(self, in_modalities):
        
        #calculating the tensor product
        
        in_modalities = [self.embed(modality) if len(modality.shape) == 2 \
                       else modality for modality in in_modalities]
        batch_size=in_modalities[0].shape[0]
        
        hidden_units = [self.text_subnet(in_modalities[0]).to(self.device)]
        for i in range(self.num_modalities-1):
            hidden_units.append(self.other_subnets[i](in_modalities[i+1]).to(self.device))
        


        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
#        if audio_h.is_cuda:
#            DTYPE = torch.cuda.FloatTensor
#        else:
#            DTYPE = torch.FloatTensor
        fusion = torch.matmul(torch.cat([torch.ones(batch_size, 1).to(self.device), hidden_units[0]], dim=1),self.text_factor)
        for i in range(self.num_modalities-1):
            added_unit = torch.cat([torch.ones(batch_size, 1).to(self.device), hidden_units[i+1]], dim=1)
            fusion_dim = torch.matmul(added_unit, self.other_factors[i])
            fusion = fusion * fusion_dim
        
        
        output = torch.matmul(self.fusion_weights, fusion.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
            
        if self.use_softmax:
            output = torch.softmax(output,dim = 0)
        return output

if __name__=="__main__":
    print("This is a module and hence cannot be called directly ...")
    print("A toy sample will now run ...")
    
    inputx=Variable(torch.Tensor(numpy.zeros([32,40])),requires_grad=True)
    inputy=Variable(torch.Tensor(numpy.array(numpy.zeros([32,12]))),requires_grad=True)
    inputz=Variable(torch.Tensor(numpy.array(numpy.zeros([32,20]))),requires_grad=True)
    modalities=[inputx,inputy,inputz]
    
    fmodel=LMF([40,12,20],100)
    
    out=fmodel(modalities)
    
    print("Output")
    print(out[0])
    print("Toy sample finished ...")






