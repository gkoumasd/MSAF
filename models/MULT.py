# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F

from layers.realnn.transformer import TransformerEncoder


class MULT(nn.Module):
    def __init__(self, opt):
        """
        Construct a MulT model.
        """
        super(MULT, self).__init__()
        self.input_dims = opt.input_dims
        self.contracted_dim = opt.contracted_dim
        
        self.output_dim = opt.output_dim      
        
        if opt.embedding_enabled:
            embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
            self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=not opt.embedding_trainable)

        # self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        # self.d_l, self.d_a, self.d_v = 30, 30, 30
#        self.orig_d_l, self.orig_d_v, self.orig_d_a = tuple(self.input_dims)
        
        
#        self.d_l = self.d_v = self.d_a = self.contracted_dim

#        self.vonly = opt.vonly
#        self.aonly = opt.aonly
#        self.lonly = opt.lonly
        
        num_modalities = len(self.input_dims)
        self.num_heads = opt.num_heads
        self.layers = opt.layers
        
#        self.attn_dropouts = opt.attn_dropouts
        if type(opt.attn_dropouts) == float:
            self.attn_dropouts = [opt.attn_dropouts]
        else:
            self.attn_dropouts = [float(s) for s in opt.attn_dropouts.split(',')]
#        self.attn_dropout_l = opt.attn_dropout_l
#        self.attn_dropout_a = opt.attn_dropout_a
#        self.attn_dropout_v = opt.attn_dropout_v
        
        self.self_attn_dropout = opt.self_attn_dropout
        self.relu_dropout = opt.relu_dropout
        self.res_dropout = opt.res_dropout
        self.out_dropout = opt.out_dropout
        self.embed_dropout = opt.embed_dropout
        self.attn_mask = opt.attn_mask

#        combined_dim = self.d_l + self.d_a + self.d_v
        combined_dim = (num_modalities-1)*self.contracted_dim* num_modalities

#        self.partial_mode = self.lonly + self.aonly + self.vonly
#        if self.partial_mode == 1:
#            combined_dim = 2 * self.d_l   # assuming d_l == d_a == d_v
#        else:
#            combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        
        # 1. Temporal convolutional layers
        self.projs = nn.ModuleList([nn.Conv1d(_dim, self.contracted_dim,  kernel_size=1, \
                                              padding = 0, bias = False) for _dim in self.input_dims])
        

#        # 1. Temporal convolutional layers
#        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
#        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
#        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        
        
        # 2. Crossmodal Attentions
        self.cross_modal_trans = nn.ModuleList()
        for i in range(len(self.input_dims)):
            trans_i = nn.ModuleList()
            for j in range(len(self.input_dims)):
                if j == i:
                    trans_i_with_j = self.get_network((num_modalities-1)*self.contracted_dim,self.self_attn_dropout,layers=3)
                else:
                    trans_i_with_j = self.get_network(self.contracted_dim,self.attn_dropouts[j])
                trans_i.append(trans_i_with_j)
            self.cross_modal_trans.append(trans_i)
        
#        if self.lonly:
#            self.trans_l_with_a = self.get_network(self_type='la')
#            self.trans_l_with_v = self.get_network(self_type='lv')
#        if self.aonly:
#            self.trans_a_with_l = self.get_network(self_type='al')
#            self.trans_a_with_v = self.get_network(self_type='av')
#        if self.vonly:
#            self.trans_v_with_l = self.get_network(self_type='vl')
#            self.trans_v_with_a = self.get_network(self_type='va')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        #self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        #self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        #self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, self.output_dim)


    def get_network(self, embed_dim, attn_dropout, layers = -1):
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
#    def get_network(self, self_type='l', layers=-1):
#        if self_type in ['l', 'al', 'vl']:
#            embed_dim, attn_dropout = self.d_l, self.attn_dropout_l
#        elif self_type in ['a', 'la', 'va']:
#            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
#        elif self_type in ['v', 'lv', 'av']:
#            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
#
#        elif self_type == 'l_mem':
#            embed_dim, attn_dropout = 2*self.d_l, self.self_attn_dropout
#
#        elif self_type == 'a_mem':
#            embed_dim, attn_dropout = 2*self.d_a, self.self_attn_dropout
#
#        elif self_type == 'v_mem':
#            embed_dim, attn_dropout = 2*self.d_v, self.self_attn_dropout
#        else:
#            raise ValueError("Unknown network type")
#        
#        return TransformerEncoder(embed_dim=embed_dim,
#                                  num_heads=self.num_heads,
#                                  layers=max(self.layers, layers),
#                                  attn_dropout=attn_dropout,
#                                  relu_dropout=self.relu_dropout,
#                                  res_dropout=self.res_dropout,
#                                  embed_dropout=self.embed_dropout,
#                                  attn_mask=self.attn_mask)
            
    def forward(self, in_modalities):

        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        
        in_modalities = [self.embed(modality) if len(modality.shape) == 2 \
                       else modality for modality in in_modalities]
#        x_l = in_modalities[0]
#        x_v = in_modalities[1]
#        x_a = in_modalities[2]
#
#        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
#        x_a = x_a.transpose(1, 2)
#        x_v = x_v.transpose(1, 2)
        
        in_modalities = [p.transpose(1,2) for p in in_modalities]
       
        # Project the textual/visual/audio features

        proj_x = [ x if self.input_dims[i] == self.contracted_dim else self.projs[i](x) for i,x in enumerate(in_modalities)]
        proj_x = [x.permute(2,0,1) for x in proj_x]
#        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
#        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
#        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
#        proj_x_a = proj_x_a.permute(2, 0, 1)
#        proj_x_v = proj_x_v.permute(2, 0, 1)
#        proj_x_l = proj_x_l.permute(2, 0, 1)
        all_hs = []
        for i in range(len(proj_x)):
            hs = []
            for j in range(len(proj_x)):
                if not i == j:
                    hs.append(self.cross_modal_trans[i][j](proj_x[i],proj_x[j],proj_x[j]))
            
            hs = torch.cat(hs, dim=2)
            hs = self.cross_modal_trans[i][i](hs)
            if type(hs) == tuple:
                hs = hs[0]
            all_hs.append(hs[-1])
                
                
#        if self.lonly:
#            # (V,A) --> L
#            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
#            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
#            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
#            h_ls = self.trans_l_mem(h_ls)
#            if type(h_ls) == tuple:
#                h_ls = h_ls[0]
#            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction
#
#        if self.aonly:
#            # (L,V) --> A
#            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
#            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
#            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
#            h_as = self.trans_a_mem(h_as)
#            if type(h_as) == tuple:
#                h_as = h_as[0]
#            last_h_a = last_hs = h_as[-1]
#            
#        if self.vonly:
#            # (L,A) --> V
#            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
#            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
#            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
#            h_vs = self.trans_v_mem(h_vs)
#            if type(h_vs) == tuple:
#                h_vs = h_vs[0]
#            last_h_v = last_hs = h_vs[-1]
        last_hs = torch.cat(all_hs, dim=1)

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output