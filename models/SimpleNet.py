# -*- coding: utf-8 -*-
import torch
from torch import nn


class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate, output_dim, output_activation = None,device = torch.device('cpu')):
        super(SimpleNet, self).__init__()
        if output_activation == None:
            self.fc = nn.Sequential(nn.Linear(input_dim,hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(hidden_dim, output_dim)
                                        )
        else:
            self.fc = nn.Sequential(nn.Linear(input_dim,hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(hidden_dim, output_dim),
                                        output_activation
                                        )
    def forward(self, input_dim):
        output = self.fc(input_dim)
        return output