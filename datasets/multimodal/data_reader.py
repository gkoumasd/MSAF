# -*- coding: utf-8 -*-

from __future__ import division
from torch.utils.data import DataLoader, TensorDataset
from preprocessor.dictionary import Dictionary
from preprocessor.embedding import Embedding
from utils.tensor import clean_tensor
import pickle
import torch
import os
import numpy as np

class MMDataReader(object):
    def __init__(self,opt):
        self.embedding_enabled = False
        self.use_sentiment_dic = False     
        self.all_feature_names = ['textual','visual','acoustic']
        self.feature_indexes = [self.all_feature_names.index(f.strip()) for f in opt.features.split(',')]                
        self.data_path = os.path.join(opt.pickle_dir_path, opt.dataset_name)+'_data.pkl'
        self.loss_weights = torch.FloatTensor([
                                        1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668,
                                        ])
    def read(self, opt):  
        
        for key,value in opt.__dict__.items():
            if not key == 'feature_indexes':
                self.__setattr__(key,value) 
        X_train, X_test, X_dev, y_train, y_test, y_dev = self.load_pickle_data()
        
        if len(X_train[0].shape) == len(X_train[1].shape)-1 and not self.embedding_enabled:
            raise Exception('Error - Embedding not enabled!')
            
        self.datas = {
                    'train':{'X': X_train,'y':y_train},
                    'test':{'X':X_test,'y':y_test},
                    'dev':{'X':X_dev,'y':y_dev}                   
                        }    
         
        self.train_sample_num = len(X_train[0])
        
        if self.label == 'sentiment' or self.dataset_name == 'avec':
            self.output_dim = 1
        
        elif self.dialogue_format:
            self.output_dim = y_train.shape[-1]
        else:
            for shape in y_train.shape[1:]:
                self.output_dim = self.output_dim*shape
            
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
        opt.sentiment_dic = self.sentiment_dic
        opt.loss_weights = self.loss_weights
        if 'embedding' in self.__dict__:
            opt.lookup_table = self.embedding.lookup_table
        opt.speaker_num = self.speaker_num
        if 'emotion_dic' in self.__dict__:
            opt.emotion_dic = self.emotion_dic
        
    def get_max_seq_len(self,features):
        if 'max_seq_len' not in self.__dict__:
            self.max_seq_len = max([len(feature) for feature in features])
            
    def flatten(self,data):
        textual_features = []
        visual_features = []
        acoustic_features = []
        labels =[]
            
        if self.dataset_name == 'meld':
            for t_dialogue,a_dialogue,label_dialogue in zip(data['text'],data['audio'],data[self.label]):
                labels.extend(label_dialogue)
                for t_sen,a_sen in zip(t_dialogue,a_dialogue):
                    textual_features.append(self.embedding.text_to_sequence(t_sen))
                    # Only sentence level feature is present
                    if type(a_sen) == np.ndarray and a_sen.ndim == 1:
                        a_sen = np.tile(a_sen,(len(t_sen),1))      
                    acoustic_features.append(a_sen)
            
            X = [textual_features, acoustic_features]
            y = labels
        else:
            for t_dialogue,v_dialogue,a_dialogue,label_dialogue in zip(data['text'],data['vision'],data['audio'],data[self.label]):
                labels.extend(label_dialogue)
                for t_sen,v_sen,a_sen in zip(t_dialogue,v_dialogue,a_dialogue):
                    textual_features.append(self.embedding.text_to_sequence(t_sen))
                    # Only sentence level feature is present
                    if type(v_sen) == np.ndarray and v_sen.ndim == 1:
                        v_sen = np.tile(v_sen,(len(t_sen),1))
                    if type(a_sen) == np.ndarray and a_sen.ndim == 1:
                        a_sen = np.tile(a_sen,(len(t_sen),1))
                    
                    visual_features.append(v_sen)
                    acoustic_features.append(a_sen)
            
            X = [textual_features, visual_features, acoustic_features]
            y = labels
        return X,y 
        
    def extract_context_info(self, X,y):
        
        context_len = self.context_len
        textual_features, visual_features,acoustic_features,speaker_features,masks = X
        queries_t = []
        queries_v = []
        queries_a = []
        own_contexts_t = []
        own_contexts_v = []
        own_contexts_a = []
        other_contexts_t = []
        other_contexts_v = []
        other_contexts_a = []
        own_speaker_masks = []
        other_speaker_masks = []
        labels = []
        
        for dia_t,dia_v,dia_a,dia_speaker,dia_mask,dia_y in zip(textual_features, visual_features,acoustic_features, speaker_features,masks,y):
            i = 0
            for sen_t, sen_v, sen_a, sen_speaker,sen_mask,sen_y in zip(dia_t,dia_v,dia_a,dia_speaker,dia_mask,dia_y):
                # Judge if it is a valid sentence or merely a zero-padded sequence
                if sen_mask == 1:
                    own_context_t = [np.zeros_like(sen_t)]* context_len
                    own_context_v = [np.zeros_like(sen_v)]* context_len
                    own_context_a = [np.zeros_like(sen_a)]* context_len
                    other_context_t = [np.zeros_like(sen_t)]* context_len
                    other_context_v = [np.zeros_like(sen_v)]* context_len
                    other_context_a = [np.zeros_like(sen_a)]* context_len
                    own_speaker_mask = np.zeros(context_len)
                    other_speaker_mask = np.zeros(context_len)
                    for j in range(context_len):
                        if i-(context_len-j)>=0:
                            if np.array_equal(sen_speaker, dia_speaker[i-(context_len-j)]):
                                own_context_t[j] = dia_t[i-(context_len-j)]
                                own_context_v[j] = dia_v[i-(context_len-j)]
                                own_context_a[j] = dia_a[i-(context_len-j)] 
                                own_speaker_mask[j] = 1
                            else:
                                other_context_t[j] = dia_t[i-(context_len-j)]
                                other_context_v[j] = dia_v[i-(context_len-j)]
                                other_context_a[j] = dia_a[i-(context_len-j)]
                                other_speaker_mask[j] = 1
                         
                    queries_t.append(np.asarray(sen_t))
                    queries_v.append(np.asarray(sen_v))
                    queries_a.append(np.asarray(sen_a))
                    own_contexts_t.append(np.asarray(own_context_t))
                    own_contexts_v.append(np.asarray(own_context_v))
                    own_contexts_a.append(np.asarray(own_context_a))
                    other_contexts_t.append(np.asarray(other_context_t))
                    other_contexts_v.append(np.asarray(other_context_v))
                    other_contexts_a.append(np.asarray(other_context_a))
                    own_speaker_masks.append(np.asarray(own_speaker_mask))
                    other_speaker_masks.append(np.asarray(other_speaker_mask))
                    labels.append(sen_y)
                i = i+1
        X = [queries_t, queries_v, queries_a, own_contexts_t,own_contexts_v,own_contexts_a, \
             other_contexts_t, other_contexts_v,other_contexts_a,own_speaker_masks, other_speaker_masks]
        y = labels
            
        return X,y
            
    def pad_dialogue(self, data):
        textual_features = []
        visual_features = []
        acoustic_features = []
        speaker_features = []
        masks = []
        labels = []
           
        for t_dialogue, v_dialogue, a_dialogue, speaker_dialogue, label_dialogue in zip(data['language'], data['vision'], data['audio']\
                                                                        , data['speaker_ids'],data[self.label]):
            
            if not type(t_dialogue) == list:
                t_dialogue = t_dialogue.tolist()
            
            if not type(v_dialogue) == list:
                v_dialogue = v_dialogue.tolist()
            
            if not type(a_dialogue) == list:
                a_dialogue = a_dialogue.tolist()
                
            textual_features.append(np.concatenate([t_dialogue+ [np.zeros_like(t_dialogue[0])]*(self.max_seq_len - len(t_dialogue))]))
            visual_features.append(np.concatenate([v_dialogue+ [np.zeros_like(v_dialogue[0])]*(self.max_seq_len - len(v_dialogue))]))
            acoustic_features.append(np.concatenate([a_dialogue+ [np.zeros_like(a_dialogue[0])]*(self.max_seq_len - len(a_dialogue))]))
            s_dialogue = np.zeros((self.max_seq_len,self.speaker_num))

            for i,_id in enumerate(speaker_dialogue):
                s_dialogue[i,_id]= 1
            speaker_features.append(s_dialogue)
            mask_dialogue = np.zeros((self.max_seq_len))
            mask_dialogue[:len(t_dialogue)] = 1
            masks.append(mask_dialogue)
            labels.append(np.concatenate([label_dialogue+ [np.zeros_like(label_dialogue[0])]*(self.max_seq_len - len(label_dialogue))]))                
        X = [textual_features,visual_features,acoustic_features,speaker_features,masks]
        y = labels
        
        return X,y
            
    def load_pickle_data(self):
        data = pickle.load(open(self.data_path, 'rb'))
        X_train_path = os.path.join(self.pickle_dir_path, self.dataset_name)+'_train.pkl'
        X_test_path = os.path.join(self.pickle_dir_path, self.dataset_name)+'_test.pkl'
        X_dev_path = os.path.join(self.pickle_dir_path, self.dataset_name)+'_valid.pkl'
        if 'emotion_dic' in data and self.label == 'emotion':
            self.emotion_dic = data['emotion_dic']
            
        self.get_max_seq_len(data['train']['text']+data['test']['text']+data['valid']['text'])
        if self.dialogue_format:
            self.speaker_num = data['speaker_num']

            if not os.path.exists(X_train_path):
                print("Creating new train data!")
                X_train, y_train = self.pad_dialogue(data['train'])
                if self.dialogue_context:
                    X_train, y_train = self.extract_context_info(X_train, y_train)
                pickle.dump([*X_train, y_train],open(X_train_path,'wb'))

            else:
                print("  - Found cached train data")
                train_data = pickle.load(open(X_train_path,'rb'))
                X_train = train_data[:-1]
                y_train = train_data[-1]
                
            if not os.path.exists(X_test_path):
                print("Creating new test data!")
                X_test, y_test = self.pad_dialogue(data['test'])
                if self.dialogue_context:
                    X_test, y_test = self.extract_context_info(X_test, y_test)
                pickle.dump([*X_test, y_test],open(X_test_path,'wb'))

            else:
                print("  - Found cached test data")
                test_data = pickle.load(open(X_test_path,'rb'))
                X_test = test_data[:-1]
                y_test = test_data[-1]
                
            if not os.path.exists(X_dev_path):
                print("Creating new dev data!")
                X_dev, y_dev = self.pad_dialogue(data['valid'])
                if self.dialogue_context:
                    X_dev, y_dev = self.extract_context_info(X_dev, y_dev)
                pickle.dump([*X_dev, y_dev],open(X_dev_path,'wb'))

            else:
                print("  - Found cached dev data")
                dev_data = pickle.load(open(X_dev_path,'rb'))
                X_dev = dev_data[:-1]
                y_dev = dev_data[-1]
            
                
            X_train = [torch.tensor(x,dtype = torch.float32) for x in X_train]
            X_test = [torch.tensor(x,dtype = torch.float32) for x in X_test]
            X_dev = [torch.tensor(x,dtype = torch.float32) for x in X_dev]
            self.embedding_enabled = False
            self.sentiment_dic = None
            
        else:
            embedding_path = os.path.join(self.pickle_dir_path, self.dataset_name)+'_embedding.pkl'     
            if not os.path.exists(embedding_path):
                print("Creating new embeddings!")
                self.dictionary = Dictionary(start_feature_id = 0)
                self.dictionary.add('UNK')
                textual_features = data['train']['text']+data['test']['text']+data['valid']['text']
                for dialogue in textual_features:
                    for tokens in dialogue:
                        for token in tokens:
                            self.dictionary.add(str(token))
            
                self.embedding = Embedding(self.dictionary,self.max_seq_len)                              
                self.embedding.get_embedding(dataset_name = self.dataset_name, fname=self.wordvec_path)
                pickle.dump(self.embedding, open(embedding_path,'wb'))
                        
            else:     
                print("  - Found cached embeddings")              
                self.embedding = pickle.load(open(embedding_path,'rb'))
                
            self.sentiment_dic = None
            if self.use_sentiment_dic == True:
                self.sentiment_dic = self.build_sentiment_lexicon()
                
            if not os.path.exists(X_train_path):
                print("Creating new train data!")
                X_train, y_train = self.flatten(data['train'])
                pickle.dump([*X_train, y_train],open(X_train_path,'wb'))
            else:
                print("  - Found cached train data")
                train_data = pickle.load(open(X_train_path,'rb'))
                X_train = train_data[:-1]
                y_train = train_data[-1]
                
            if not os.path.exists(X_test_path):
                print("Creating new test data!")
                X_test, y_test = self.flatten(data['test'])
                pickle.dump([*X_test, y_test],open(X_test_path,'wb'))
            else:
                print("  - Found cached test data")
                test_data = pickle.load(open(X_test_path,'rb'))
                X_test = test_data[:-1]
                y_test = test_data[-1]
                
            if not os.path.exists(X_dev_path):
                print("Creating new valid data!")
                X_dev, y_dev = self.flatten(data['valid'])
                pickle.dump([*X_dev, y_dev],open(X_dev_path,'wb'))
            else:
                print("  - Found cached valid data")
                dev_data = pickle.load(open(X_dev_path,'rb'))
                X_dev = dev_data[:-1]
                y_dev = dev_data[-1]
            
            X_train = [torch.tensor(x,dtype = torch.int64) if i==0 else torch.tensor(x,dtype = torch.float32) for i,x in enumerate(X_train)]           
            X_test = [torch.tensor(x,dtype = torch.int64) if i==0 else torch.tensor(x,dtype = torch.float32) for i,x in enumerate(X_test)]
            X_dev = [torch.tensor(x,dtype = torch.int64) if i==0 else torch.tensor(x,dtype = torch.float32) for i,x in enumerate(X_dev)]
            self.embedding_enabled = True
            self.speaker_num = 1

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
        
    def get_train(self, shuffle=True, iterable=True):
        x = self.datas['train']['X']
        y = self.datas['train']['y']
        feature_indexes = [_ind for _ind in self.feature_indexes]
        
        # Always include textual modality 
        if 0 not in feature_indexes:
            feature_indexes = [0]+feature_indexes
            
        if self.dialogue_format:
            # If dialogue, include speaker ids and dialogue masks 
            
            if self.dialogue_context:
                num_modalities = int((len(x)-2)/3)
                index_set_1 = [_id + num_modalities for _id in feature_indexes]
                index_set_2 = [_id + 2*num_modalities for _id in feature_indexes]

                feature_indexes.extend(index_set_1)
                feature_indexes.extend(index_set_2)
               
            feature_indexes.extend([len(x)-2,len(x)-1])
        
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
            
    def get_test(self, shuffle=True, iterable=True):
        x = self.datas['test']['X']
        y = self.datas['test']['y']
        feature_indexes = [_ind for _ind in self.feature_indexes]
        # Always include textual modality 
        if 0 not in feature_indexes:
            feature_indexes = [0]+feature_indexes
            
        if self.dialogue_format:
            # If dialogue, include speaker ids and dialogue masks 
            if self.dialogue_context:
                num_modalities = int((len(x)-2)/3)
                index_set_1 = [_id + num_modalities for _id in feature_indexes]
                index_set_2 = [_id + 2*num_modalities for _id in feature_indexes]

                feature_indexes.extend(index_set_1)
                feature_indexes.extend(index_set_2)
               
            feature_indexes.extend([len(x)-2,len(x)-1])

        x = [_x for i,_x in enumerate(x) if i in feature_indexes]
        
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
        feature_indexes = [_ind for _ind in self.feature_indexes]
        # Always include textual modality 
        if 0 not in feature_indexes:
            feature_indexes = [0]+feature_indexes
        if self.dialogue_format:
            
            # If dialogue, include speaker ids and dialogue masks 
            if self.dialogue_context:
                num_modalities = int((len(x)-2)/3)
                index_set_1 = [_id + num_modalities for _id in feature_indexes]
                index_set_2 = [_id + 2*num_modalities for _id in feature_indexes]

                feature_indexes.extend(index_set_1)
                feature_indexes.extend(index_set_2)
               
            feature_indexes.extend([len(x)-2,len(x)-1])
            
        x = [_x for i,_x in enumerate(x) if i in feature_indexes]
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