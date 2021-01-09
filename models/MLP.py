# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class MLP(nn.Module): 
    def __init__(self, opt): 
        super(MLP, self).__init__() 
        self.input_dims = opt.input_dims
        if opt.embedding_enabled:
            embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
            self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.fc_out = nn.Sequential(nn.Linear(sum(self.input_dims), opt.hidden_dim_1),
                                    nn.ReLU(),
                                    nn.Dropout(opt.dropout_rate_1),
                                    nn.Linear(opt.hidden_dim_1,opt.hidden_dim_2),
                                    nn.ReLU(),
                                    nn.Dropout(opt.dropout_rate_1),
                                    nn.Linear(opt.hidden_dim_2,opt.output_dim))

    def forward(self, in_modalities):
        in_modalities = [self.embed(modality) if len(modality.shape) == 2 \
                       else modality for modality in in_modalities]
        avg_features = torch.cat([torch.mean(modality,dim = 1) for modality in in_modalities],dim = -1)
        output = self.fc_out(avg_features)
        return output
