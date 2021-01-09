# -*- coding: utf-8 -*-

from __future__ import division
from torch.utils.data import DataLoader, TensorDataset
from preprocessor.dictionary import Dictionary
from preprocessor.embedding import Embedding
from utils.generic import clean_tensor
import pickle
import torch
import os

class CMUMOSEIReader(object):
    def __init__(self,opt):
        self.embedding_enabled = True
        #self.use_sentiment_dic = False     
        self.all_feature_names = ['textual','visual','acoustic']
        self.feature_indexes = [self.all_feature_names.index(f.strip()) for f in opt.features.split(',')]                
        self.fp_prefix = os.path.join(opt.pickle_dir_path, 'cmumosei')
        self.data_path = self.fp_prefix +'_data.pkl'

    def read(self, opt):  
        
        for key,value in opt.__dict__.items():
            if not key == 'feature_indexes':
                self.__setattr__(key,value) 
        X_train, X_test, X_dev, y_train, y_test, y_dev = self.load_pickle_data()
            
        self.datas = {
                    'train':{'X': X_train,'y':y_train},
                    'test':{'X':X_test,'y':y_test},
                    'dev':{'X':X_dev,'y':y_dev}                   
                        }    
         
        self.train_sample_num = len(X_train[0])
        self.output_dim = 1            
        self.input_dims = [x.shape[-1] for i,x in enumerate(X_train) if (i in self.feature_indexes)]
          
        if self.embedding_enabled:
            self.input_dims[0] = self.embedding.embedding_size
            
        self.opt_callback(opt)
        
    def opt_callback(self,opt):
        opt.dataset_name = self.dataset_name
        opt.feature_indexes = self.feature_indexes
        opt.input_dims = self.input_dims
        opt.train_sample_num = self.train_sample_num
        opt.output_dim = self.output_dim
        opt.embedding_enabled = self.embedding_enabled
        opt.max_seq_len = self.max_seq_len
        #opt.sentiment_dic = self.sentiment_dic
        if 'embedding' in self.__dict__:
            opt.lookup_table = self.embedding.lookup_table
        #opt.speaker_num = self.speaker_num
        if 'emotion_dic' in self.__dict__:
            opt.emotion_dic = self.emotion_dic
        
    def get_max_seq_len(self,features):
        if 'max_seq_len' not in self.__dict__:
            self.max_seq_len = max([len(feature) for feature in features])
            
    def load_pickle_data(self):
        data = pickle.load(open(self.data_path, 'rb'))
        X_train_path = self.fp_prefix+'_train.pkl'
        X_test_path = self.fp_prefix+'_test.pkl'
        X_dev_path = self.fp_prefix+'_valid.pkl'
            
        self.get_max_seq_len(data['train']['text']+data['test']['text']+data['valid']['text'])
        
        # Load embedding
        embedding_path = self.fp_prefix+'_embedding.pkl'     
        if not os.path.exists(embedding_path):
            print("Creating new embeddings!")
            self.dictionary = Dictionary(start_feature_id = 0)
            self.dictionary.add('UNK')
            textual_features = data['train']['text']+data['test']['text']+data['valid']['text']
            for tokens in textual_features:
                for token in tokens:
                    self.dictionary.add(str(token.lower()))
        
            self.embedding = Embedding(self.dictionary,self.max_seq_len)                              
            self.embedding.get_embedding(dataset_name = self.dataset_name, fname=self.wordvec_path)
            pickle.dump(self.embedding, open(embedding_path,'wb'))
                    
        else:     
            print("  - Found cached embeddings")              
            self.embedding = pickle.load(open(embedding_path,'rb'))
            
        if not os.path.exists(X_train_path):
            print("Creating new train data!")
            X_train = [[self.embedding.text_to_sequence(seq) for seq in data['train']['text']], data['train']['vision'], data['train']['audio']]
            y_train = data['train'][self.label]
            pickle.dump([*X_train, y_train],open(X_train_path,'wb'))
        else:
            print("  - Found cached train data")
            train_data = pickle.load(open(X_train_path,'rb'))
            X_train = train_data[:-1]
            y_train = train_data[-1]

        if not os.path.exists(X_test_path):
            print("Creating new test data!")
            X_test = [[self.embedding.text_to_sequence(seq) for seq in data['test']['text']], data['test']['vision'], data['test']['audio']]
            y_test  = data['test'][self.label]
            pickle.dump([*X_test, y_test],open(X_test_path,'wb'))
        else:
            print("  - Found cached test data")
            test_data = pickle.load(open(X_test_path,'rb'))
            X_test = test_data[:-1]
            y_test = test_data[-1]
            
        if not os.path.exists(X_dev_path):
            print("Creating new valid data!")
            X_dev = [[self.embedding.text_to_sequence(seq) for seq in data['valid']['text']],  data['valid']['vision'], data['valid']['audio']]
            y_dev  = data['valid'][self.label]
            pickle.dump([*X_dev, y_dev],open(X_dev_path,'wb'))
        else:
            print("  - Found cached valid data")
            dev_data = pickle.load(open(X_dev_path,'rb'))
            X_dev = dev_data[:-1]
            y_dev = dev_data[-1]
        
        
        # Convert data to tensor format
        X_train = [torch.tensor(x,dtype = torch.int64) if i==0 else torch.tensor(x,dtype = torch.float32) for i,x in enumerate(X_train)]           
        X_test = [torch.tensor(x,dtype = torch.int64) if i==0 else torch.tensor(x,dtype = torch.float32) for i,x in enumerate(X_test)]
        X_dev = [torch.tensor(x,dtype = torch.int64) if i==0 else torch.tensor(x,dtype = torch.float32) for i,x in enumerate(X_dev)]

        # Remove spurious values (-inf)
        for x in X_train:
            clean_tensor(x)
        for x in X_test:
            clean_tensor(x)
        for x in X_dev:
            clean_tensor(x)


        y_train = torch.tensor(y_train,dtype = torch.float32)
        y_test = torch.tensor(y_test,dtype = torch.float32)
        y_dev = torch.tensor(y_dev,dtype = torch.float32)
        
        if y_train.dim() == 3:
            y_train = y_train.squeeze(dim = -1)
            y_test = y_test.squeeze(dim = -1)
            y_dev = y_dev.squeeze(dim = -1)

        return X_train, X_test, X_dev, y_train, y_test, y_dev
        
    def get_data(self, shuffle=True, iterable=True, split='train'):
        x = self.datas[split]['X']
        y = self.datas[split]['y']
        feature_indexes = [_ind for _ind in self.feature_indexes]
        
        # Always include textual modality 
        if 0 not in feature_indexes:
            feature_indexes = [0]+feature_indexes
        
        x = [_x for i,_x in enumerate(x) if i in feature_indexes]
        
        if iterable:
            all_tensors = []
            for _x in x:
                all_tensors.append(_x)
            all_tensors.append(y)
            trainDataset = TensorDataset(*all_tensors)
            train_loader = DataLoader(trainDataset, batch_size = self.batch_size, shuffle = shuffle)
            return train_loader
        else:
            return x,y
        
        