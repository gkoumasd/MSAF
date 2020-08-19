# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from mmsdk import mmdatasdk
from preprocessor.dictionary import Dictionary
from preprocessor.embedding import Embedding
from utils.const import FEATURE_NAME_DIC
from utils.tensor import clean_tensor
import pickle
import torch
import os
import pandas as pd

class MMDataReader(object):
    def __init__(self,dir_path, feature_name_dic, standard_folds, opt):
        
        self.embedding_enabled = False
        for key,value in opt.__dict__.items():
            self.__setattr__(key,value)  
        
        self.all_feature_names = ['textual','visual','acoustic']
        self.feature_indexes = [self.all_feature_names.index(f.strip()) for f in opt.features.split(',')]                

        if self.load_data_from_pickle == True:
            X_train, X_test, X_dev, y_train, y_test, y_dev = self.load_pickle_data()
            
        else:        
            X_train, X_test, X_dev, y_train, y_test, y_dev = self.read_data_from_sdk(dir_path,feature_name_dic, standard_folds, opt)
        
        if len(X_train[0].shape) == len(X_train[1].shape)-1 and not self.embedding_enabled:
            raise Exception('Error - Embedding not enabled!')
            
        self.datas = {
                    'train':{'X': X_train,'y':y_train},
                    'test':{'X':X_test,'y':y_test},
                    'dev':{'X':X_dev,'y':y_dev}                   
                        }             
        self.train_sample_num = len(X_train[0])
        self.output_dim = 1
        for shape in y_train.shape[1:]:
            self.output_dim = self.output_dim*shape
            
        self.max_seq_len = X_train[0].shape[1]            
        self.input_dims = [x.shape[-1] for i,x in enumerate(X_train) if (i in self.feature_indexes)]

        if self.embedding_enabled:
            self.input_dims[0] = self.embedding.embedding_size
        self.opt_callback(opt)
        
    
        
        
        
    def load_pickle_data(self):

        data_path = os.path.join(self.pickle_dir_path, self.dataset_name)+'_data.pkl'
        data = pickle.load(open(data_path, 'rb'))
        X_train_path = os.path.join(self.pickle_dir_path, self.dataset_name)+'_train.pkl'
        X_test_path = os.path.join(self.pickle_dir_path, self.dataset_name)+'_test.pkl'

        X_dev_path = os.path.join(self.pickle_dir_path, self.dataset_name)+'_valid.pkl'
        
        # for CMU SDK Data, the boolean variable embedding_enabled is also stored in the pickle file
        if len(data) == 2:
            self.embedding_enabled = data[1]
            data = data[0]

        if self.embedding_enabled:
            embedding_path = os.path.join(self.pickle_dir_path, self.dataset_name)+'_embedding.pkl'     
            if not os.path.exists(embedding_path):
                print("Creating new embeddings!")
                self.dictionary = Dictionary(start_feature_id = 0)
                self.dictionary.add('UNK')
                textual_features = data['train']['text']+data['test']['text']+data['valid']['text']
                for tokens in textual_features:
                    for token in tokens:
                        self.dictionary.add(str(token))
            
                self.embedding = Embedding(self.dictionary,self.max_seq_len)                              
                self.embedding.get_embedding(dataset_name = self.dataset_name, fname=self.wordvec_path)
                pickle.dump(self.embedding, open(embedding_path,'wb'))
                
            else:     
                print("  - Found cached embeddings")              
                self.embedding = pickle.load(open(embedding_path,'rb'))
                
            data['train']['text'] = [self.embedding.text_to_sequence(x) for x in data['train']['text']]
            data['test']['text'] = [self.embedding.text_to_sequence(x) for x in data['test']['text']]
            data['valid']['text'] = [self.embedding.text_to_sequence(x) for x in data['valid']['text']]
            
            
        if not os.path.exists(X_train_path):
            print("Creating new train data!")
            X_train = [data['train']['text'],data['train']['vision'],data['train']['audio']]
            y_train = data['train']['labels']
            pickle.dump([*X_train, y_train],open(X_train_path,'wb'))
        else:
            print("  - Found cached train data")
            train_data = pickle.load(open(X_train_path,'rb'))
            X_train = train_data[:-1]
            y_train = train_data[-1]
            
        if not os.path.exists(X_test_path):
            print("Creating new test data!")
            X_test = [data['test']['text'],data['test']['vision'],data['test']['audio']]
            y_test = data['test']['labels']
            pickle.dump([*X_test, y_test],open(X_test_path,'wb'))
        else:
            print("  - Found cached test data")
            test_data = pickle.load(open(X_test_path,'rb'))
            X_test = test_data[:-1]
            y_test = test_data[-1]
            
        if not os.path.exists(X_dev_path):
            print("Creating new valid data!")
            X_dev = [data['valid']['text'],data['valid']['vision'],data['valid']['audio']]
            y_dev = data['valid']['labels']
            pickle.dump([*X_dev, y_dev],open(X_dev_path,'wb'))
        else:
            print("  - Found cached valid data")
            dev_data = pickle.load(open(X_dev_path,'rb'))
            X_dev = dev_data[:-1]
            y_dev = dev_data[-1]
            

        if self.embedding_enabled:            
            X_train = [torch.tensor(x,dtype = torch.int64) if i==0 else torch.tensor(x,dtype = torch.float32) for i,x in enumerate(X_train)]           
            X_test = [torch.tensor(x,dtype = torch.int64) if i==0 else torch.tensor(x,dtype = torch.float32) for i,x in enumerate(X_test)]
            X_dev = [torch.tensor(x,dtype = torch.int64) if i==0 else torch.tensor(x,dtype = torch.float32) for i,x in enumerate(X_dev)]
        else:
            X_train = [torch.tensor(x,dtype = torch.float32) for x in X_train]
            X_test = [torch.tensor(x,dtype = torch.float32) for x in X_test]
            X_dev = [torch.tensor(x,dtype = torch.float32) for x in X_dev]
            
            
        
            
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
        
        
        y_train = self.categorical_labels(y_train)
        y_test = self.categorical_labels(y_test)
        y_dev = self.categorical_labels(y_dev)
        
        
        
        if y_train.dim() == 3:
            y_train = y_train.squeeze(dim = -1)
            y_test = y_test.squeeze(dim = -1)
            y_dev = y_dev.squeeze(dim = -1)
        
        
        return X_train, X_test, X_dev, y_train, y_test, y_dev

    def categorical_labels(self,data):
        positive = data>=0
        negative = data<0
        
        return torch.tensor(torch.cat((positive, negative), -1),dtype=torch.float32)
        
    

    def read_data_from_sdk(self, dir_path, feature_name_dic, standard_folds, opt):
        dataset = mmdatasdk.mmdataset(dir_path)
        #Load all features
        features = [] 
        for f in self.all_feature_names:
            feature_name = feature_name_dic[f.strip()][0]
            features.append(dataset.computational_sequences[feature_name].data)
        
        #Load classes
        label_name = feature_name_dic[opt.label][0]
        classes = dataset.computational_sequences[label_name].data
        
        #Load data splits
        self.video_id_train = standard_folds.standard_train_fold
        self.video_id_test = standard_folds.standard_test_fold
        self.video_id_dev = standard_folds.standard_valid_fold
                 
        # Filter out the videos that are not shared by all features   
        self.get_common_ids(classes,features)
                   
        # Determine the maximum sequence length
        self.get_max_seq_len(features)
        
        # Get word embedding if word pivots are present
        self.get_embedding(features)
       
        # Process and normalize dataset
        video_label_dic, video_feature_dics,video_interval_dic = self.process_data(features,classes)
        
        # Split train/test/dev data
        X_train, X_test, X_dev, y_train, y_test, y_dev, id_train, id_test, id_dev, interval_train, interval_test, interval_dev= self.split_data(video_label_dic, video_feature_dics, video_interval_dic)       
        
        # Textual modality is in the inputs
        if 0 in self.feature_indexes:
            self.embedding_enabled = True
        
        print('save data to pickle file:')
        train_data = {'text': X_train[0],'vision': X_train[1],'audio': X_train[2], 'labels': y_train,'ids': id_train,'intervals': interval_train}
        test_data = {'text': X_test[0],'vision': X_test[1],'audio': X_test[2], 'labels': y_test,'ids': id_test,'intervals': interval_test}
        dev_data = {'text': X_dev[0],'vision': X_dev[1],'audio': X_dev[2], 'labels': y_dev,'ids': id_dev,'intervals': interval_dev}
        data = {'train':train_data, 'test':test_data, 'valid':dev_data}
        pickle.dump([data,self.embedding_enabled],open(os.path.join(opt.datasets_dir,'cmusdk_data','{}_data.pkl'.format(opt.dataset_name)),'wb'))
        pickle.dump(self.embedding, open(os.path.join(opt.datasets_dir,'cmusdk_data','{}_embedding.pkl'.format(opt.dataset_name)),'wb'))
        print('Done.')
        
        return X_train, X_test, X_dev, y_train, y_test, y_dev
            
   
        # Normalize visual features
#            if self.normalize_visual_feature and not visual_id == -1:       
#                X_train, X_test, X_dev = self.normalize_visual(X_train, X_test, \
#                                                               X_dev, visual_id)                
    def opt_callback(self,opt):
        opt.feature_indexes = self.feature_indexes
        opt.input_dims = self.input_dims
        opt.train_sample_num = self.train_sample_num
        opt.output_dim = self.output_dim
        opt.embedding_enabled = self.embedding_enabled
        opt.max_seq_len = self.max_seq_len
        if self.embedding_enabled:
            opt.lookup_table = self.embedding.lookup_table
        
    def get_train(self, shuffle=True, iterable=True):
        x = self.datas['train']['X']
        y = self.datas['train']['y']
        x = [_x for i,_x in enumerate(x) if i in self.feature_indexes]

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
            
    def get_test(self, shuffle=True, iterable=True):
        x = self.datas['test']['X']
        y = self.datas['test']['y']
        x = [_x for i,_x in enumerate(x) if i in self.feature_indexes]

        if iterable:
            all_tensors = []
            for _x in x:
                all_tensors.append(_x)
            all_tensors.append(y)
            trainDataset = TensorDataset(*all_tensors)
            train_loader = DataLoader(trainDataset, batch_size = self.batch_size, shuffle = True)
            return train_loader
        else:
            return x,y
    
    def get_val(self, shuffle=True, iterable=True):
        x = self.datas['dev']['X']
        y = self.datas['dev']['y']
        x = [_x for i,_x in enumerate(x) if i in self.feature_indexes]

        if iterable:
            all_tensors = []
            for _x in x:
                all_tensors.append(_x)
            all_tensors.append(y)
            trainDataset = TensorDataset(*all_tensors)
            train_loader = DataLoader(trainDataset, batch_size = self.batch_size, shuffle = True)
            return train_loader
        else:
            return x,y
        
    def setup(self,opt):
         for key,value in opt.__dict__.items():
            self.__setattr__(key,value)  
            
            
    def normalize_visual(self, X_train, X_test, X_dev, visual_id):
        visual_train = X_train[visual_id]
        visual_test = X_test[visual_id]
        visual_dev = X_dev[visual_id]
        visual_max = [torch.max(torch.abs(visual_train[:,:,i])) for i in range(self.input_dims[visual_id])]
        for i in range(self.input_dims[visual_id]):
            X_train[visual_id][:,:,i] = visual_train[:,:,i]/visual_max[i]
            X_test[visual_id][:,:,i] = visual_test[:,:,i]/visual_max[i]
            X_dev[visual_id][:,:,i] = visual_dev[:,:,i]/visual_max[i]
        return X_train, X_test, X_dev
            
           
    def get_max_seq_len(self,features):
        if 'max_seq_len' not in self.__dict__:
            length_list = []
            for video_id in self.common_video_ids:
                length_list.append(features[0][video_id]['features'].value.shape[0])
            length_list = np.asarray(length_list)
            self.max_seq_len = int(np.std(length_list) + np.mean(length_list))
        
    def get_max_dialogue_len(self, video_label_dic):
        dialogue_len_list = []
        for vid in video_label_dic.keys():
            dialogue_len_list.append(len(video_label_dic[vid]))
                                  
        dialogue_len_list = np.asarray(dialogue_len_list)
        self.max_dialogue_len = int(np.std(dialogue_len_list) + np.mean(dialogue_len_list))
        
    def pad_sequence(self, seq):
        new_seq = []
        seq_type = type(seq[0])
        if seq_type == int:
            seq[0] = [seq[0]]
            seq_type = list
        if seq_type == list:    
            for sample in seq:
                sample = [sample] if not type(sample) == list else sample
                if len(sample) >=self.max_seq_len:
                    new_seq.append(sample[:self.max_seq_len])
                else:
                    new_seq.append(sample + [0]* (self.max_seq_len-len(sample)))
        elif seq_type == np.ndarray:
            for sample in seq:
                if sample.shape[0]>=self.max_seq_len:
                    s = sample[:self.max_seq_len,:]
                else:
                    s = np.concatenate([sample, np.zeros((self.max_seq_len-len(sample),sample.shape[-1]))])
                new_seq.append(s)
                
        if self.dialogue_format:
            if len(new_seq) >= self.max_dialogue_len:
                new_seq = new_seq[:self.max_dialogue_len]
            else:
                if seq_type == list:
                    new_seq = new_seq+ [[0]*self.max_seq_len]*(self.max_dialogue_len-len(new_seq))
                elif seq_type == np.ndarray:
                    new_seq = new_seq + [np.zeros((self.max_seq_len,new_seq[0].shape[-1]))]*(self.max_dialogue_len-len(new_seq))
                
        return new_seq
    
    def get_embedding(self,features):
        self.dictionary = Dictionary(start_feature_id = 0)
        self.dictionary.add('sp')
        textual_features = features[0]
        for video_id in textual_features:
            tokens = textual_features[video_id]['features'].value
            for token in tokens:
                self.dictionary.add(str(token[0],'utf-8'))
            
        self.embedding = Embedding(self.dictionary,self.max_seq_len)                              
        self.embedding.get_embedding(dataset_name = self.dataset_name, fname=self.wordvec_path)
            
    
    def get_common_ids(self, classes, features):
        self.common_video_ids = classes.keys()
        for one_modality in features:
            self.common_video_ids = self.common_video_ids & one_modality.keys()
            
    def process_label(self, classes):
        all_classes = []
        for key in classes:
            all_classes.append(classes[key]['features'].value[0][0])
        unique_classes = list(set(all_classes))
        
        label_dict = {}
        if self.one_hot_label == False:
            for class_label in unique_classes:
                l = np.zeros((1))
                l[0] = class_label
                label_dict[class_label] = l
        else:
            index = 0
            for class_label in unique_classes:          
                one_hot_indexes = np.zeros((len(unique_classes)))
                one_hot_indexes[index] = 1
                label_dict[class_label] = one_hot_indexes
                index = index+1
        
        self.label_dict = label_dict
        
    def process_data(self,features,classes):
        self.process_label(classes)
        video_label_dic = {}
        video_feature_dics = [{} for i in range(len(self.all_feature_names))]    
        video_interval_dic = {}        

        for video_id in self.common_video_ids:
            vid = video_id.split('[')[0]
            sub_vid = int(video_id.split('[')[1][:-1])
            l = classes[video_id]['features'].value[0][0]
            l = self.label_dict[l]
            interval = classes[video_id]['intervals'].value[0]
            if vid not in video_label_dic:
                video_label_dic[vid] = [np.zeros_like(l)]*(sub_vid+1)
                video_label_dic[vid][sub_vid] = l
                video_interval_dic[vid] = [np.zeros_like(interval)]*(sub_vid+1)
                video_interval_dic[vid][sub_vid] = interval

            else:
                label_list = video_label_dic[vid]
                if sub_vid >=len(label_list):
                    label_list = label_list+[np.zeros_like(l)]*(sub_vid-len(label_list)+1)                              
                label_list[sub_vid] = l
                video_label_dic[vid] = label_list
                
                interval_list = video_interval_dic[vid]
                if sub_vid >=len(interval_list):
                    interval_list = interval_list+[np.zeros_like(interval)]*(sub_vid-len(interval_list)+1)                              
                interval_list[sub_vid] = interval
                video_interval_dic[vid] = interval_list
                
            
            for i, one_modality in enumerate(features):   
                f = one_modality[video_id]['features'].value 
                if i == 0:
                    f = [str(s[0],'utf-8') for s in f]
                    f = self.embedding.text_to_sequence(f)
                    if not type(f) == list:
                        f = [f]
                else:
                    f[f!=f] = 0
                    f[f == float('-inf')] = 0      
                    if f.shape[0] > self.max_seq_len:
                        f = f[:self.max_seq_len,:]
                        
                dic = video_feature_dics[i]
                if vid not in dic:
                    if type(f) == list:
                        dic[vid] = [0*self.max_seq_len]*(sub_vid+1)
                    else:
                        dic[vid] = [np.zeros_like(f)]*(sub_vid+1)
                            
                    dic[vid][sub_vid] = f
                else:
                    feature_list = dic[vid]
                    if sub_vid < len(feature_list):
                        feature_list[sub_vid] = f
                    else:
                        if type(f) == list:
                            feature_list = feature_list+ [[0]*self.max_seq_len]*(sub_vid-len(feature_list)+1)
                        elif type(f) == np.ndarray:
                            feature_list = feature_list+ [np.zeros_like(f)]*(sub_vid-len(feature_list)+1)

                        feature_list[sub_vid] = f
                    dic[vid] = feature_list
        return video_label_dic, video_feature_dics, video_interval_dic
    
    def split_data(self, video_label_dic, video_feature_dics, video_interval_dic):
        X_train = [[] for i in range(len(self.all_feature_names))]
        X_test = [[] for i in range(len(self.all_feature_names))]
        X_dev = [[] for i in range(len(self.all_feature_names))]
        y_train = []
        y_test = []
        y_dev = []
        id_train = []
        id_test = []
        id_dev = []
        interval_train = []
        interval_test = []
        interval_dev = []
        
        if self.dialogue_format:
            self.get_max_dialogue_len(video_label_dic)
            
        for vid in video_label_dic.keys():
            label_list = video_label_dic[vid]  
            interval_list = video_interval_dic[vid]
            if self.dialogue_format:
                if len(label_list)>= self.max_dialogue_len:
                    label_list = label_list[:self.max_dialogue_len]
                label_list = label_list+ [np.zeros_like(label_list[0])]*(self.max_dialogue_len-len(label_list))
            
            if vid in self.video_id_train:
                y_train.append(torch.tensor(label_list,dtype = torch.float32))
                for i in range(len(label_list)):
                    id_train.append(vid + '[{}]'.format(i))                
                    interval_train.append(interval_list[i])
                    
            elif vid in self.video_id_test:
                y_test.append(torch.tensor(label_list,dtype = torch.float32))
                for i in range(len(label_list)):
                    id_test.append(vid + '[{}]'.format(i))
                    interval_test.append(interval_list[i])

            elif vid in self.video_id_dev:
                y_dev.append(torch.tensor(label_list,dtype = torch.float32))
                for i in range(len(label_list)):
                    id_dev.append(vid + '[{}]'.format(i))
                    interval_dev.append(interval_list[i])
                    
#        dic = video_feature_dics[0]
#        for vid in video_label_dic.keys(): 
            
            for i, dic in enumerate(video_feature_dics):
                f = dic[vid]
                f = self.pad_sequence(f)
                f_type = torch.float32
                if i == 0:
                    f_type = torch.int64
                    
                if vid in self.video_id_train:
                    X_train[i].append(torch.tensor(f,dtype = f_type))
                elif vid in self.video_id_test:
                    X_test[i].append(torch.tensor(f,dtype = f_type))
                elif vid in self.video_id_dev:
                    X_dev[i].append(torch.tensor(f,dtype = f_type))
        X_train =[torch.stack(x) for x in X_train] if self.dialogue_format else [torch.cat(x) for x in X_train]
        X_dev = [torch.stack(x) for x in X_dev] if self.dialogue_format else [torch.cat(x) for x in X_dev]
        X_test = [torch.stack(x) for x in X_test] if self.dialogue_format else [torch.cat(x) for x in X_test]
        
        y_train = torch.stack(y_train) if self.dialogue_format else torch.cat(y_train)
        y_dev = torch.stack(y_dev) if self.dialogue_format else torch.cat(y_dev)
        y_test = torch.stack(y_test) if self.dialogue_format else torch.cat(y_test)
        
        return X_train, X_test, X_dev, y_train, y_test, y_dev, id_train, id_test, id_dev, interval_train, interval_test, interval_dev

          
        
class CMUMOSEIDataReader(MMDataReader):
    def __init__(self,dir_path, opt):     
        from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSEI import standard_folds
        self.one_hot_label = False
        super(CMUMOSEIDataReader,self).__init__(dir_path,FEATURE_NAME_DIC['cmumosei'], standard_folds,opt)
        
class CMUMOSIDataReader(MMDataReader):
    def __init__(self,dir_path, opt):      
        from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSI import standard_folds
        self.one_hot_label = False
        super(CMUMOSIDataReader,self).__init__(dir_path,FEATURE_NAME_DIC['cmumosi'],standard_folds,opt)
        
class POMDataReader(MMDataReader):
    def __init__(self,dir_path, opt):      
        from mmsdk.mmdatasdk.dataset.standard_datasets.POM import standard_folds
        self.one_hot_label = False
        super(POMDataReader,self).__init__(dir_path,FEATURE_NAME_DIC['pom'],standard_folds,opt)
        
class IEMOCAPDataReader(MMDataReader):
    def __init__(self,dir_path, opt):      
        from mmsdk.mmdatasdk.dataset.standard_datasets.IEMOCAP import standard_folds
        self.one_hot_label = True
        super(IEMOCAPDataReader,self).__init__(dir_path,FEATURE_NAME_DIC['iemocap'],standard_folds,opt)
    


        
        
        
        
        
        
        
        
