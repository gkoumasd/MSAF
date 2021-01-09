# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
import os
import copy
from utils.evaluation import evaluate
import time
import pickle
from optimizer import RMSprop_Unitary
from utils.params import Params

def train(params, model):
    criterion = get_criterion(params)
    if hasattr(model,'get_params'):
        unitary_params, remaining_params = model.get_params()
    else:
        remaining_params = model.parameters()
        unitary_params = []
        
    if len(unitary_params)>0:
        unitary_optimizer = RMSprop_Unitary(unitary_params,lr = params.unitary_lr)

    #remaining_parameters = get_remaining_parameters(model,unitary_parameters)
    optimizer = torch.optim.RMSprop(remaining_params,lr = params.lr)  

    # Temp file for storing the best model 
    temp_file_name = str(int(np.random.rand()*int(time.time())))
    params.best_model_file = os.path.join('tmp',temp_file_name)

    best_val_loss = 99999.0
    for i in range(params.epochs):
        print('epoch: ', i)
        model.train()
        with tqdm(total = params.train_sample_num) as pbar:
            time.sleep(0.05)            
            for _i,data in enumerate(params.reader.get_data(iterable = True, shuffle = True,split = 'train'),0):
#                For debugging, please run the line below
                #_i,data = next(iter(enumerate(params.reader.get_data(iterable = True, shuffle = True),0)))

                b_inputs = [inp.to(params.device) for inp in data[:-1]]
                b_targets = data[-1].to(params.device)
                
                optimizer.zero_grad()
                outputs = model(b_inputs)
                
                # IEMOCAP
                if not outputs.shape == b_targets.shape:
                    outputs = outputs.reshape_as(b_targets)

                loss = get_loss(params, criterion, outputs, b_targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), params.clip)
                optimizer.step()
                
                if len(unitary_params)>0:
                    unitary_optimizer.step()
                    
                # Compute Training Accuracy                                  
                n_total = len(outputs)
                if params.label == 'sentiment':
                    n_correct = (outputs.sign() == b_targets.sign()).sum().item()
                    
                elif params.label == 'emotion': 
                    #outputs: (n_total, num_classes, 2)
                    #b_targets: (n_total, num_classes, 2)
                    num_classes = b_targets.shape[-2]
                    n_correct = (outputs.argmax(dim = -1) == b_targets.argmax(dim = -1)).sum().item()/num_classes
                train_acc = n_correct/n_total 

                #Update Progress Bar
                pbar.update(params.batch_size)
                ordered_dict={'acc': train_acc, 'loss':loss.item()}        
                pbar.set_postfix(ordered_dict=ordered_dict)
        
        model.eval()
        
        #################### Compute Validation Performance##################
        val_output, val_target = get_predictions(model, params, split = 'dev')
             
        if params.label == 'emotion': 
            val_output = val_output.reshape_as(val_target)
            
        val_loss = get_loss(params, criterion, val_output, val_target)
        
        print('validation performance:')
        performances = evaluate(params,val_output,val_target)        
        
        print('val_acc = {}, val_loss = {}'.format(performances['acc'], val_loss))
        ##################################################################        
        
        if val_loss < best_val_loss:
            torch.save(model,params.best_model_file)
            print('The best model up till now. Saved to File.')
            best_val_loss = val_loss

def get_criterion(params):
    # Only 1-dim output, regression loss is used
    # For monologue sentiment regression
    if params.output_dim == 1:
        criterion = nn.L1Loss()
    else:
        criterion = nn.NLLLoss()
        
    return criterion

def get_loss(params, criterion, outputs, b_targets):
    # Only 1-dim output, regression loss is used
    # For monologue sentiment regression
    if params.output_dim == 1:
        loss = criterion(outputs,b_targets)

    # Multi-class classification
    # For IEMOCAP
    else:
        #outputs: (n_total, num_classes, 2)
        #b_targets: (n_total, num_classes, 2)
        log_outputs = F.log_softmax(outputs,dim = -1)
        loss = criterion(log_outputs.reshape(-1, 2),b_targets.argmax(dim=-1).reshape(-1))
    return loss

def test(model,params):
    model.eval()
    test_output,test_target= get_predictions(model, params, split = 'test')    
    
    if params.label == 'emotion': 
        test_output = test_output.reshape_as(test_target)      
         
    performances = evaluate(params,test_output,test_target)
    return performances

def get_predictions(model, params, split ='dev'):
    outputs = []
    targets = []
    iterator = params.reader.get_data(iterable =True, shuffle = False, split = split)
        
    for _ii,data in enumerate(iterator,0):  
        data_x = [inp.to(params.device) for inp in data[:-1]]
        data_t = data[-1].to(params.device)
        data_o = model(data_x)
        outputs.append(data_o.detach())
        targets.append(data_t.detach())
            
    outputs = torch.cat(outputs)
    targets = torch.cat(targets)
        
    return outputs, targets

#Save the model
def save_model(model,params,performance_str):
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
        
    #Create model dir
    params.dir_name = str(round(time.time()))
    dir_path = os.path.join('tmp',params.dir_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        
    #save the learned model
    torch.save(model.state_dict(),os.path.join(dir_path,'model'))
    params.export_to_config(os.path.join(dir_path,'config.ini'))
    
    #save the configuration file
    temp_params = copy.deepcopy(params)
    if 'lookup_table' in temp_params.__dict__:
        del temp_params.lookup_table
    if 'sentiment_dic' in temp_params.__dict__:
        del temp_params.sentiment_dic
    del temp_params.reader
    pickle.dump(temp_params, open(os.path.join(dir_path,'config.pkl'),'wb'))
    del temp_params
    
#    if 'save_phases' in params.__dict__ and params.save_phases:
#        print('Saving Phases.')
#        phase_dict = model.get_phases()
#        for key in phase_dict:
#            file_path = os.path.join(dir_path,'{}_phases.pkl'.format(key))
#            pickle.dump(phase_dict[key],open(file_path,'wb'))
    
    #Write performance string
    eval_path = os.path.join(dir_path,'eval')
    with open(eval_path,'w') as f:
        f.write(performance_str)
    
def print_performance(performance_dict, params):
    performance_str = ''
    if params.label == 'sentiment' or params.dialogue_format:
        for key, value in performance_dict.items():
            performance_str = performance_str+ '{} = {} '.format(key,value)
    elif params.label == 'emotion':
        performance_str = performance_str +'acc = {}\n'.format(performance_dict['acc'])
        emotions = ["Neutral", "Happy", "Sad", "Angry"]
        acc_per_class = performance_dict['acc_per_class']
        f1_per_class = performance_dict['f1_per_class']
        for i in range(4):
            performance_str = performance_str + '{}: acc = {}, f1 = {}\n'.format(emotions[i],acc_per_class[i],f1_per_class[i])
    print(performance_str)
    return performance_str

def print_result_from_dir(dir_path):
    params = Params()
    params.parse_config(os.path.join(dir_path, 'config.ini'))
    reader = open(os.path.join(dir_path,'eval'),'r')
    s = reader.readline().split()
    print('dataset: {}, network_type: {}, acc: {}, f1:{}'.format(params.dataset_name,params.network_type,s[2],s[5]))
    
def save_performance(params, performance_dict):
        
    df = pd.DataFrame()
    output_dic = {'dataset' : params.dataset_name,
                    'modality' : params.features,
                    'network' : params.network_type,
                    'model_dir_name': params.dir_name}
    output_dic.update(performance_dict)
    df = df.append(output_dic, ignore_index = True)

    if not 'output_file' in params.__dict__:
        params.output_file = 'eval/{}_{}.csv'.format(params.dataset_name, params.network_type)
    df.to_csv(params.output_file, encoding='utf-8', index=True)


