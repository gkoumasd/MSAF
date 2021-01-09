# -*- coding: utf-8 -*-
"""
Created on Tue Mar 5 11:44:48 2019


"""
import torch
import random

import os
import models
import argparse
import pandas as pd
import pickle
from dataset import setup 
from utils.model import train,test,save_model,save_performance,print_performance
from utils.io import parse_grid_parameters
from utils.generic import set_seed
from utils.params import Params


def run(params):   
    model = None
    if 'load_model_from_dir' in params.__dict__ and params.load_model_from_dir:
        print('Loading the model from an existing dir!')
        model_params = pickle.load(open(os.path.join(params.dir_name,'config.pkl'),'rb'))
        if 'lookup_table' in params.__dict__:
            model_params.lookup_table = params.lookup_table
        if 'sentiment_dic' in params.__dict__:
            model_params.sentiment_dic = params.sentiment_dic
        model = models.setup(model_params)
        model.load_state_dict(torch.load(os.path.join(params.dir_name,'model')))
        model = model.to(params.device)
    else:
        model = models.setup(params).to(params.device)
      
    if not ('fine_tune' in params.__dict__ and params.fine_tune == False):
        print('Training the model!')
        train(params, model)
        model = torch.load(params.best_model_file)
        os.remove(params.best_model_file)
    
    performance_dict = test(model, params)
    performance_str = print_performance(performance_dict, params)
    save_model(model,params,performance_str)
  
    return performance_dict

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='running experiments on multimodal datasets.')
    parser.add_argument('-config', action = 'store', dest = 'config_file', help = 'please enter configuration file.',default = 'config/run.ini')
    args = parser.parse_args()
    params = Params()
    params.parse_config(args.config_file) 
    params.config_file = args.config_file
    mode = 'run'
    if 'mode' in params.__dict__:
        mode = params.mode
    set_seed(params)
    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if mode == 'run':
        results = []
        reader = setup(params)
        reader.read(params)
        params.reader = reader   
        performance_dict = run(params)
        save_performance(params, performance_dict)
       
    elif mode == 'run_grid_search':
            
        print('Grid Search Begins.')
        if not 'grid_parameters_file' in params.__dict__:
            params.grid_parameters_file = params.network_type+'.ini'
            
        grid_parameters = parse_grid_parameters(os.path.join('config','grid_parameters',params.grid_parameters_file))
        df = pd.DataFrame()
        if not 'output_file' in params.__dict__:
            params.output_file = 'eval/grid_search_{}_{}.csv'.format(params.dataset_name, params.network_type)
        for i in range(params.search_times):
            parameter_list = []
            merged_dict = {}
            for key in grid_parameters:
                value = random.choice(grid_parameters[key])
                parameter_list.append((key, value))
                merged_dict[key] = value
            print(parameter_list)
            params.setup(parameter_list)
            reader = setup(params)
            reader.read(params)
            params.reader = reader
            performance_dict = run(params)
            performance_dict['model_dir_name'] = params.dir_name
            merged_dict.update(performance_dict)
            df = df.append(merged_dict, ignore_index=True)      
            df.to_csv(params.output_file, encoding='utf-8', index=True)
    else:
        print('wrong input run mode!')
        exit(1)
        
        
        
   
