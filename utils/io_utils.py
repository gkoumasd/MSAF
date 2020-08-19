# -*- coding: utf-8 -*-
import h5py
import os
import configparser
import re
import numpy as np
def load_saved_data(dir_path):
    h5f = h5py.File(os.path.join(dir_path,'X_train.h5'),'r')
    X_train = h5f['data'][:]
    h5f.close()
    h5f = h5py.File(os.path.join(dir_path,'y_train.h5'),'r')
    y_train = h5f['data'][:]
    h5f.close()
    h5f = h5py.File(os.path.join(dir_path,'X_valid.h5'),'r')
    X_valid = h5f['data'][:]
    h5f.close()
    h5f = h5py.File(os.path.join(dir_path,'y_valid.h5'),'r')
    y_valid = h5f['data'][:]
    h5f.close()
    h5f = h5py.File(os.path.join(dir_path,'X_test.h5'),'r')
    X_test = h5f['data'][:]
    h5f.close()
    h5f = h5py.File(os.path.join(dir_path,'y_test.h5'),'r')
    y_test = h5f['data'][:]
    h5f.close()
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def load_selected_features(file_path):
        f = open(file_path, 'r')
        indexes = [ int(s.strip()) for s in f.read().strip().split('\t')]
        return indexes
    
def parse_grid_parameters(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    config_common = config['COMMON']
    dictionary = {}
    for key,value in config_common.items():
        array = value.split(';')
        is_numberic = re.compile(r'^[-+]?[0-9.]+$')
        new_array = []
    
        for value in array:
            value = value.strip()
            result = is_numberic.match(value)
            if result:
                if type(eval(value)) == int:
                    value= int(value)
                else :
                    value= float(value)
            new_array.append(value)
        dictionary[key] = new_array
    return dictionary

def load_raw_data(dir_path):
    h5f = h5py.File(os.path.join(dir_path,'audio_train.h5'),'r')
    audio_train = h5f['data']
    h5f.close()
    h5f = h5py.File(os.path.join(dir_path,'audio_test.h5'),'r')
    audio_test = h5f['data']
    h5f.close()
    h5f = h5py.File(os.path.join(dir_path,'audio_valid.h5'),'r')
    audio_valid = h5f['data']
    h5f.close()
#    h5f = h5py.File(os.path.join(dir_path,'y_valid.h5'),'r')
#    y_valid = h5f['data'][:]
#    h5f.close()
#    h5f = h5py.File(os.path.join(dir_path,'X_test.h5'),'r')
#    X_test = h5f['data'][:]
#    h5f.close()
#    h5f = h5py.File(os.path.join(dir_path,'y_test.h5'),'r')
#    y_test = h5f['data'][:]
#    h5f.close()
    return audio_train, audio_test, audio_valid
    
def to_one_hot(input_array):
    output = np.zeros((input_array.size, input_array.max()+1))
    output[np.arange(input_array.size),input_array] = 1
    return output

def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]
#dir_path = 'data/cmumosei_old'
#h5f = h5py.File(os.path.join(dir_path,'y_train.h5'),'r')
#a_group_key = list(h5f.keys())[0]
#data = list(h5f[a_group_key])
#print(data)
#print(len(data),data[0].shape)
