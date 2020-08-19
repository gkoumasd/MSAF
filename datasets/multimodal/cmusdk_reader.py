# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from mmsdk import mmdatasdk
from utils.const import FEATURE_NAME_DIC
import pickle
import os

class MMDataReader(object):
    def __init__(self, feature_name_dic, standard_folds, opt):
        
        self.dataset_dir_path = os.path.join(opt.datasets_dir,opt.data_dir)
        self.feature_name_dic = feature_name_dic
        self.standard_folds = standard_folds
        self.embedding_enabled = False
        self.use_sentiment_dic = False
        
        for key,value in opt.__dict__.items():
            self.__setattr__(key,value)  
        
        self.feature_name_pairs = {'textual':'text','visual':'vision','acoustic':'audio'}

    def read_data_from_sdk(self):
        dataset = mmdatasdk.mmdataset(self.dataset_dir_path)
        
        self.speakers = ['none']
        if self.dataset_name == 'iemocap':
            self.speakers = self.feature_name_dic['speaker_markers']
        #Load all features
        features = [] 
        saved_feature_names = []
        for f, _name in self.feature_name_pairs.items():
            feature_name = self.feature_name_dic[f.strip()][0]
            features.append(dataset.computational_sequences[feature_name].data)
            saved_feature_names.append(_name)
        
        #Load classes
        self.label_names = self.label.split(',')
        classes = []
        for _name in self.label_names:
            label_name = self.feature_name_dic[_name][0]
            classes.append(dataset.computational_sequences[label_name].data)
        
        #Load data splits
        self.video_id_train = self.standard_folds.standard_train_fold
        self.video_id_test = self.standard_folds.standard_test_fold
        self.video_id_dev = self.standard_folds.standard_valid_fold
                 
        # Filter out the videos that are not shared by all features   
        self.get_common_ids(classes[0],features)
                   
        if self.data_aligned:
            # Determine the maximum sequence length
            self.get_max_seq_len(features)
           
            # Process and normalize dataset
            video_label_dics, video_feature_dics,video_interval_dic,video_speaker_dic = self.process_data(features,classes)
            
            # Split train/test/dev data
            X_train, X_test, X_dev, y_train, y_test, y_dev, id_train, id_test, id_dev, interval_train, interval_test, interval_dev, speaker_train, speaker_test, speaker_dev= self.split_data(video_label_dics, video_feature_dics, video_interval_dic,video_speaker_dic)       
             
            print('save data to pickle file:')
            train_data = {saved_feature_names[0]: X_train[0],saved_feature_names[1]: X_train[1],saved_feature_names[2]: X_train[2], 'ids': id_train,'intervals': interval_train,'speaker_ids':speaker_train}
            test_data = {saved_feature_names[0]: X_test[0],saved_feature_names[1]: X_test[1],saved_feature_names[2]: X_test[2], 'ids': id_test,'intervals': interval_test,'speaker_ids':speaker_test}
            dev_data = {saved_feature_names[0]: X_dev[0],saved_feature_names[1]: X_dev[1],saved_feature_names[2]: X_dev[2], 'ids': id_dev,'intervals': interval_dev,'speaker_ids':speaker_dev}
            emotion_dic = None
            for i, _name in enumerate(self.label_names):
                train_data[_name] = y_train[i]
                test_data[_name] = y_test[i]
                dev_data[_name] = y_dev[i]
                if _name == 'emotion':
                    emotion_dic = self.label_dicts[i]
            data = {'train':train_data, 'test':test_data, 'valid':dev_data,'speaker_num':len(self.speakers)}
            if emotion_dic is not None:
                data['emotion_dic'] = emotion_dic
            pickle.dump(data,open(os.path.join(self.datasets_dir,'cmusdk_data','{}_{}_{}.pkl'.format(self.dataset_name, '_'.join(self.label_names), self.max_seq_len)),'wb'))
            print('Done.')
        else:
            X_train, X_train_interval, X_test, X_test_interval, X_dev, X_dev_interval,\
                y_train,y_train_interval, y_test, y_test_interval, y_dev, y_dev_interval,id_train, id_test, id_dev\
                = self.process_unaligned_data(features,classes)
            print('save data to pickle file:')
            train_data = {saved_feature_names[0]: X_train[0],saved_feature_names[0]+'_intervals': X_train_interval[0],\
                          saved_feature_names[1]: X_train[1],saved_feature_names[1]+'_intervals': X_train_interval[1],\
                          saved_feature_names[2]: X_train[2],saved_feature_names[2]+'_intervals': X_train_interval[2],\
                          'labels': y_train,'labels_intervals': y_train_interval, 'ids': id_train}
            test_data = {saved_feature_names[0]: X_test[0],saved_feature_names[0]+'_intervals': X_test_interval[0],\
                          saved_feature_names[1]: X_test[1],saved_feature_names[1]+'_intervals': X_test_interval[1],\
                          saved_feature_names[2]: X_test[2],saved_feature_names[2]+'_intervals': X_test_interval[2],\
                          'labels': y_test,'labels_intervals': y_test_interval, 'ids': id_test}
            dev_data = {saved_feature_names[0]: X_dev[0],saved_feature_names[0]+'_intervals': X_dev_interval[0],\
                          saved_feature_names[1]: X_dev[1],saved_feature_names[1]+'_intervals': X_dev_interval[1],\
                          saved_feature_names[2]: X_dev[2],saved_feature_names[2]+'_intervals': X_dev_interval[2],\
                          'labels': y_dev,'labels_intervals': y_dev_interval, 'ids': id_dev}
            data = {'train':train_data, 'test':test_data, 'valid':dev_data,'speaker_num':len(self.speakers)}
            pickle.dump(data,open(os.path.join(self.datasets_dir,'cmusdk_data','{}_{}_unaligned.pkl'.format(self.dataset_name, self.label)),'wb'))
            print('Done.')
                
            
    def get_max_seq_len(self,features):
        if 'max_seq_len' not in self.__dict__:
            length_list = []
            for video_id in self.common_video_ids:
                length_list.append(features[0][video_id]['features'].value.shape[0])
            length_list = np.asarray(length_list)
            self.max_seq_len = int(np.std(length_list) + np.mean(length_list))
    
    def get_common_ids(self, classes, features):
        self.common_video_ids = classes.keys()
        for one_modality in features:
            self.common_video_ids = self.common_video_ids & one_modality.keys()
            
    def process_label(self, classes):
        self.label_dicts = []
        for _class in classes:
            all_classes = []
            for key in _class:
                l = _class[key]['features'][()][0][0]
                if type(l) == np.bytes_:
                    if not str(l,'utf-8')[-1].isalpha():
                        l = l[:-1]
                all_classes.append(l)
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
        
            self.label_dicts.append(label_dict)
    
    def process_unaligned_data(self,features,classes):  
        X_train = [[] for i in range(len(self.feature_name_pairs))]
        X_test = [[] for i in range(len(self.feature_name_pairs))]
        X_dev = [[] for i in range(len(self.feature_name_pairs))]
        y_train = []
        y_train_interval = []
        y_test = []
        y_test_interval = []
        y_dev = []
        y_dev_interval = []

        id_train = []
        id_test = []
        id_dev = []
        X_train_interval = [[] for i in range(len(self.feature_name_pairs))]
        X_test_interval = [[] for i in range(len(self.feature_name_pairs))]
        X_dev_interval = [[] for i in range(len(self.feature_name_pairs))]
        
        for vid in self.common_video_ids:
            label_list = classes[vid]['features'].value
            label_interval_list = classes[vid]['intervals'].value
            if vid in self.video_id_train:
                y_train.append(label_list)
                y_train_interval.append(label_interval_list)
                id_train.append(vid)                
#                interval_train.append(interval_list)
                
            elif vid in self.video_id_test:
                y_test.append(label_list)
                y_test_interval.append(label_interval_list)
                id_test.append(vid) 
#                interval_test.append(interval_list)

            elif vid in self.video_id_dev:
                y_dev.append(label_list)
                y_dev_interval.append(label_interval_list)
                id_dev.append(vid) 
#                interval_dev.append(interval_list)
            
            for i, dic in enumerate(features):
                f = dic[vid]['features'].value
                if i == 0:
                    f = [str(s[0],'utf-8') for s in f]
                    f = ['UNK' if token =='sp' else token for token in f]

                interval_list = dic[vid]['intervals'].value    
                if vid in self.video_id_train:
                    X_train[i].append(f)
                    X_train_interval[i].append(interval_list)
                elif vid in self.video_id_test:
                    X_test[i].append(f)
                    X_test_interval[i].append(interval_list)

                elif vid in self.video_id_dev:
                    X_dev[i].append(f)
                    X_dev_interval[i].append(interval_list)
        return X_train, X_train_interval, X_test, X_test_interval, X_dev, X_dev_interval,\
                y_train,y_train_interval, y_test, y_test_interval, y_dev, y_dev_interval,id_train, id_test, id_dev
    
    def get_speaker_info(self,classes):
        self.speaker_ids = {}
        for key in self.common_video_ids:
            _id = 0
            l = classes[0][key]['features'][()][0][0]
            if type(l) == np.bytes_:
                if not str(l,'utf-8')[-1].isalpha():
                     _id = self.speakers.index(l[-1])
            
            self.speaker_ids[key] = _id
        
        
    def process_data(self,features,classes):
        self.get_speaker_info(classes)
        self.process_label(classes)
        video_label_dics = [{} for i in range(len(classes))]
        video_feature_dics = [{} for i in range(len(self.feature_name_pairs))]    
        video_interval_dic = {}        
        video_speaker_dic = {}        

        for video_id in self.common_video_ids:
            vid = video_id.split('[')[0]
            sub_vid = int(video_id.split('[')[1][:-1])
            
            #Intervals
            interval = classes[0][video_id]['intervals'].value[0]
            speaker_id = self.speaker_ids[video_id]

            if vid not in video_label_dics[0]:
                video_interval_dic[vid] = [np.zeros_like(interval)]*(sub_vid+1)
                video_interval_dic[vid][sub_vid] = interval
                video_speaker_dic[vid] = [-1]*(sub_vid+1)
                video_speaker_dic[vid][sub_vid] = speaker_id
            else:
                interval_list = video_interval_dic[vid]
                if sub_vid >=len(interval_list):
                    interval_list = interval_list+[np.zeros_like(interval)]*(sub_vid-len(interval_list)+1)                              
                interval_list[sub_vid] = interval
                video_interval_dic[vid] = interval_list
                
                speaker_list = video_speaker_dic[vid]
                if sub_vid >=len(speaker_list):
                    speaker_list = speaker_list+[-1]*(sub_vid-len(speaker_list)+1)   
                speaker_list[sub_vid] = speaker_id
                video_speaker_dic[vid] = speaker_list
            
            #Labels
            for i, _class in enumerate(classes):
                _itv = _class[video_id]['intervals'][()]
                l =  _class[video_id]['features'][()][(_itv == interval).all(axis=1)]

                if self.label_names[i] == 'sentiment' or self.dataset_name == 'iemocap':
                    label = l[0][0]
                    if type(label) == np.bytes_:
                        if not str(label,'utf-8')[-1].isalpha():
                            label = label[:-1]
                    l = self.label_dicts[i][label]
                elif self.label_names[i] == 'emotion':
                    l = l[0]
                if vid not in video_label_dics[i]:
                    video_label_dics[i][vid] = [np.zeros_like(l)]*(sub_vid+1)
                    video_label_dics[i][vid][sub_vid] = l
    
                else:
                    label_list = video_label_dics[i][vid]
                    if sub_vid >=len(label_list):
                        label_list = label_list+[np.zeros_like(l)]*(sub_vid-len(label_list)+1)                              
                    label_list[sub_vid] = l
                    video_label_dics[i][vid] = label_list
            
            #Features
            for i, one_modality in enumerate(features):   
                f = one_modality[video_id]['features'].value 
                if i == 0:
                    f = [str(s[0],'utf-8') for s in f]
                    if self.dataset_name == 'iemocap':
                        f = [s[:-1] for s in f]
                    f = self.pad_seq(f)
                    if not type(f) == list:
                        f = [f]
                else:
                    f[f!=f] = 0
                    f[f == float('-inf')] = 0      
                    f = self.pad_seq(f)
#                    if f.shape[0] > self.max_seq_len:
#                        f = f[:self.max_seq_len]
#                    else:
#                        f = np.zeros(self.max_seq_len,)
#                        f = np.concatenate([f,np.zeros((self.max_seq_len-len(f),*f.shape[1:]))])
                        
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
        return video_label_dics, video_feature_dics, video_interval_dic, video_speaker_dic
    
    def pad_seq(self, f, pad_type = 'post', pad_token = 'UNK'):
        output = None
        if type(f) == np.ndarray:
            if len(f) > self.max_seq_len:
                output = f[:self.max_seq_len]
            else:
                zeros_array =  np.zeros((self.max_seq_len-len(f),*f.shape[1:]))
                if pad_type == 'pre':
                    output = np.concatenate([zeros_array,f])
                elif pad_type == 'post':
                    output = np.concatenate([f,zeros_array])
        else:
            str_list = f
            str_list = [pad_token if token =='sp' else token for token in str_list]
            if len(str_list) > self.max_seq_len:
                output = str_list[:self.max_seq_len]
            elif pad_type == 'pre':
                output = [pad_token] * (self.max_seq_len - len(str_list))+ str_list
            elif pad_type == 'post':
                output = str_list + [pad_token] * (self.max_seq_len - len(str_list))
        return output

    def split_data(self, video_label_dics, video_feature_dics, video_interval_dic, video_speaker_dic):
        X_train = [[] for i in range(len(self.feature_name_pairs))]
        X_test = [[] for i in range(len(self.feature_name_pairs))]
        X_dev = [[] for i in range(len(self.feature_name_pairs))]

        
        y_train = [[] for i in range(len(self.label_names))]
        y_test = [[] for i in range(len(self.label_names))]
        y_dev = [[] for i in range(len(self.label_names))]
        id_train = []
        id_test = []
        id_dev = []
        interval_train = []
        interval_test = []
        interval_dev = []
        speaker_train = []
        speaker_test = []
        speaker_dev= []
        
        for vid in video_label_dics[0].keys():
            label_list = [np.asarray(_label_dic[vid]) for _label_dic in video_label_dics]
            interval_list = video_interval_dic[vid]
            speaker_list = video_speaker_dic[vid]
            if vid in self.video_id_train:
                for k in range(len(self.label_names)):
                    y_train[k].append(label_list[k])
                id_train.append(vid) 
                speaker_train.append(speaker_list)
                interval_train.append(interval_list)
                
            elif vid in self.video_id_test:
                for k in range(len(self.label_names)):
                    y_test[k].append(label_list[k])
                id_test.append(vid) 
                speaker_test.append(speaker_list)

                interval_test.append(interval_list)

            elif vid in self.video_id_dev:
                for k in range(len(self.label_names)):
                    y_dev[k].append(label_list[k])
                id_dev.append(vid) 
                speaker_dev.append(speaker_list)
                interval_dev.append(interval_list)
            
            for i, dic in enumerate(video_feature_dics):
                
                f = dic[vid]
                
                if vid in self.video_id_train:
                    X_train[i].append(f)
                elif vid in self.video_id_test:
                    X_test[i].append(f)
                elif vid in self.video_id_dev:
                    X_dev[i].append(f)
        
        return X_train, X_test, X_dev, y_train, y_test, y_dev, id_train, id_test, id_dev, interval_train, interval_test, interval_dev, speaker_train, speaker_test, speaker_dev

          
        
class CMUMOSEIDataReader(MMDataReader):
    def __init__(self,opt):     
        from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSEI import standard_folds
        self.one_hot_label = False
        super(CMUMOSEIDataReader,self).__init__(FEATURE_NAME_DIC['cmumosei'], standard_folds,opt)
        
class CMUMOSIDataReader(MMDataReader):
    def __init__(self,opt):      
        from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSI import standard_folds
        self.one_hot_label = False
        super(CMUMOSIDataReader,self).__init__(FEATURE_NAME_DIC['cmumosi'],standard_folds,opt)
        
class POMDataReader(MMDataReader):
    def __init__(self, opt):      
        from mmsdk.mmdatasdk.dataset.standard_datasets.POM import standard_folds
        self.one_hot_label = False
        super(POMDataReader,self).__init__(FEATURE_NAME_DIC['pom'],standard_folds,opt)
        
class IEMOCAPDataReader(MMDataReader):
    def __init__(self, opt):      
        from mmsdk.mmdatasdk.dataset.standard_datasets.IEMOCAP import standard_folds
        self.one_hot_label = True
        super(IEMOCAPDataReader,self).__init__(FEATURE_NAME_DIC['iemocap'],standard_folds,opt)
    