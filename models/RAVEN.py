# -*- coding: utf-8 -*-

import torch
from torch import nn
from .SimpleNet import SimpleNet
from layers.realnn import lstm

class RAVEN(nn.Module):
    def __init__(self,opt):
        #The version for aligned data only
        super(RAVEN, self).__init__()
        
        self.device = opt.device
        self.input_dims = opt.input_dims
        self.num_modalities = len(self.input_dims)  
        if opt.embedding_enabled:
            embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
            self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=not opt.embedding_trainable)   
        self.output_dim = opt.output_dim        
 
        self.padding_len = opt.max_seq_len
        self.shift_padding_len = opt.shift_padding_len
        self.dropProb = opt.drop_prob
        self.wordDim = self.input_dims[0]
        self.covarepDim = self.input_dims[2]
        self.normDim = opt.norm_dim
        self.cellDim = opt.cell_dim
        self.facetDim = self.input_dims[1]
        self.hiddenDim = opt.hidden_dim
        self.lastState = opt.last_state
        self.layer = opt.layer
        self.shift_weight = opt.shift_weight
                 
        features = 0
        input_size = 0

        self.normcovarep = nn.BatchNorm2d(self.padding_len, track_running_stats=False)
        self.dropcovarep = nn.Dropout(p=self.dropProb)
        self.fc_rszcovarep = nn.Linear(self.covarepDim, self.normDim)
        self.covarepLSTM = nn.LSTM(self.normDim, self.cellDim)
        self.covarepW = nn.Linear(self.cellDim + self.wordDim, 1)

        self.normFacet = nn.BatchNorm2d(self.padding_len, track_running_stats=False)
        self.dropFacet = nn.Dropout(p=self.dropProb)
        self.fc_rszFacet = nn.Linear(self.facetDim, self.normDim)
        self.facetLSTM = nn.LSTM(self.normDim, self.cellDim)
        self.facetW = nn.Linear(self.cellDim + self.wordDim, 1)

        self.calcAddon = nn.Linear(2 * self.cellDim, self.wordDim)

        self.dropWord = nn.Dropout(p=self.dropProb)
        input_size += self.wordDim

        self.lstm1 = lstm.LSTM(input_size, self.hiddenDim, layer=self.layer,device = self.device)

        if self.lastState:
            self.fc_afterLSTM = nn.Linear(self.hiddenDim, 1)
        else:
            self.fc_afterLSTM = nn.Linear(self.hiddenDim * self.padding_len, 1)


        #if self.output_dim == 1:
        #    self.fc_out = SimpleNet(self.final_fc_in_dim, self.output_cell_dim,
        #                            self.out_dropout_rate,self.output_dim)
            
        #else:
        #    self.fc_out = SimpleNet(self.final_fc_in_dim, self.output_cell_dim,
        #                            self.out_dropout_rate,self.output_dim, nn.Softmax(dim = 1))

    def forward(self,in_modalities):

        in_modalities = [self.embed(modality) if len(modality.shape) == 2 \
                       else modality for modality in in_modalities]

        self.batch_size = in_modalities[0].shape[0]
        time_stamps = in_modalities[0].shape[1]

        words = in_modalities[0].to(self.device)
        covarep = torch.unsqueeze(in_modalities[2], dim = -2).to(self.device)
        covarepLens = torch.ones(self.batch_size, time_stamps, dtype = torch.long).to(self.device)
        facet = torch.unsqueeze(in_modalities[1], dim = -2).to(self.device)
        facetLens = torch.ones(self.batch_size, time_stamps, dtype = torch.long).to(self.device)
        inputLens = time_stamps*torch.ones(self.batch_size,dtype = torch.long).to(self.device)


        inputs = None
        covarep = self.normcovarep(covarep)
        covarepInput = self.fc_rszcovarep(self.dropcovarep(covarep))
        covarepFlat = covarepInput.data.contiguous().view(-1, self.shift_padding_len, self.normDim)
        output, _ = self.covarepLSTM(covarepFlat)
        output = torch.cat([torch.zeros(self.batch_size * self.padding_len, 1, self.cellDim).to(self.device), output], 1)
        covarepLensFlat = covarepLens.data.contiguous().view(-1)
        covarepSelector = torch.zeros(self.batch_size * self.padding_len, 1, self.shift_padding_len + 1).to(self.device).scatter_(2, covarepLensFlat.unsqueeze(1).unsqueeze(1), 1.0)
        covarepState = torch.matmul(covarepSelector, output).squeeze()

        #facet = self.normFacet(facet)
        facetInput = self.fc_rszFacet(self.dropFacet(facet))
        facetFlat = facetInput.data.contiguous().view(-1, self.shift_padding_len, self.normDim)
        output, _ = self.facetLSTM(facetFlat)
        output = torch.cat([torch.zeros(self.batch_size * self.padding_len, 1, self.cellDim).to(self.device), output], 1)
        facetLensFlat = facetLens.data.contiguous().view(-1)
        facetSelector = torch.zeros(self.batch_size * self.padding_len, 1, self.shift_padding_len + 1).to(self.device).scatter_(2, facetLensFlat.unsqueeze(1).unsqueeze(1), 1.0)
        facetState = torch.matmul(facetSelector, output).squeeze()

        wordFlat = words.data.contiguous().view(-1, self.wordDim)
        covarepWeight = self.covarepW(torch.cat([covarepState, wordFlat], 1))
        facetWeight = self.facetW(torch.cat([facetState, wordFlat], 1))
        covarepState = covarepState * covarepWeight
        facetState = facetState * facetWeight
        addon = self.calcAddon(torch.cat([covarepState, facetState], 1))


        addonL2 = torch.norm(addon, 2, 1)
        addonL2 = torch.max(addonL2, torch.tensor([1.0]).to(self.device)) / torch.tensor([self.shift_weight]).to(self.device)
        addon = addon / addonL2.unsqueeze(1)
        addon = addon.data.contiguous().view(self.batch_size, self.padding_len, self.wordDim)

        wordsL2 = torch.norm(words, 2, 2).unsqueeze(2)
        wordInput = self.dropWord(words + addon * wordsL2)


        inputs = wordInput

        output, _ = self.lstm1(inputs)
        if self.lastState:
            self.selector = torch.zeros(self.batch_size, 1, self.padding_len).to(self.device).scatter_(2, (inputLens-1).unsqueeze(1).unsqueeze(1), 1.0)
            spec_output = torch.matmul(self.selector, output).squeeze()
        else:
            spec_output = output.data.contiguous().view(-1, self.hiddenDim * self.padding_len)
        final = self.fc_afterLSTM(spec_output)

        return final

