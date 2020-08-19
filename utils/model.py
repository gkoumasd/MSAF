# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
import os
import copy
from utils.network import get_remaining_parameters, flatten_output, get_unitary_parameters
from shutil import copyfile
from utils.evaluation import evaluate
import time
import pickle
from optimizer import *
#from utils.tensor import clean_tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.params import Params
from utils.io_utils import powerset
from models.multimodal.dialogue.DialogueRNN import MaskedNLLLoss
def train(params, model):
    
    criterion = get_criterion(params)
    #unitary_parameters = get_unitary_parameters(model)
    if hasattr(model,'get_params'):
        unitary_params, remaining_params = model.get_params()
    else:
        remaining_params = model.parameters()
        remaining_params_int = sum(p.numel() for p in model.parameters() if p.requires_grad)
        unitary_params = []
        
         
        
    print(len(unitary_params))
    print(remaining_params_int)
        
    if len(unitary_params)>0:
        unitary_optimizer = RMSprop_Unitary(unitary_params,lr = params.unitary_lr)

    #remaining_parameters = get_remaining_parameters(model,unitary_parameters)
    optimizer = torch.optim.RMSprop(remaining_params,lr = params.lr) 
    #optimizer = torch.optim.Adam(model.parameters(),lr = params.lr)      
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=params.patience, factor=0.1, verbose=True)

    # Temp file for storing the best model 
    temp_file_name = str(int(np.random.rand()*int(time.time())))
    params.best_model_file = os.path.join('tmp',temp_file_name)

    best_val_loss = 99999.0
#    best_val_loss = -1.0
    for i in range(params.epochs):
        print('epoch: ', i)
        #break
        model.train()
        with tqdm(total = params.train_sample_num) as pbar:
            time.sleep(0.05)            
            for _i,data in enumerate(params.reader.get_train(iterable = True, shuffle = True),0):
                print(_i)
                #print(data[0].shape)
#                For debugging, please run the line below
#                _i,data = next(iter(enumerate(params.reader.get_train(iterable = True, shuffle = True),0)))

                b_inputs = [inp.to(params.device) for inp in data[:-1]]
                b_targets = data[-1].to(params.device)
                
                
                # Does not train if batch_size is 1, because batch normalization will crash
                if b_inputs[0].shape[0] == 1:
                    continue
                optimizer.zero_grad()
                if len(unitary_params)>0:
                    unitary_optimizer.zero_grad()

                outputs = model(b_inputs)
                b_targets, outputs, loss = get_loss(params, criterion, outputs, b_targets, b_inputs[-1])
                if np.isnan(loss.item()):
                    print('loss value overflow!')
                    torch.save(model,params.best_model_file)
                    i = params.epochs
                    break           
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
                    n_correct = (outputs.argmax(dim = -1) == b_targets).sum().item()
                train_acc = n_correct/n_total 

                #Update Progress Bar
                pbar.update(params.batch_size)
                ordered_dict={'acc': train_acc, 'loss':loss.item()}        
                pbar.set_postfix(ordered_dict=ordered_dict)
        
        model.eval()
        
        #################### Compute Validation Performance##################
        val_output,val_target, val_mask = get_predictions(model, params, split = 'dev')
             
        val_target, val_output, val_loss = get_loss(params, criterion,val_output,val_target, val_mask)

                
        print('validation performance:')
        performances = evaluate(params,val_output,val_target)        
        
        print('val_acc = {}, val_loss = {}'.format(performances['acc'], val_loss))
        scheduler.step(val_loss)
        torch.cuda.empty_cache()
        ##################################################################
        
        
#        test_output,test_target, test_mask = get_predictions(model, params, split = 'test')
#        
#        print('test performance:')
#        performances = evaluate(params,test_output,test_target) 
#        
#        if params.label == 'emotion':
#            _test_output = test_output.view(-1,test_output.shape[-1])
#            _test_target = test_target.view(-1,test_target.shape[-1])  
#            test_loss = criterion(_test_output,_test_target.argmax(dim = -1)).item()
#        elif params.label == 'sentiment':
#            test_loss = criterion(test_output,test_target).item()
#        print('test_acc = {}, test_loss = {}'.format(performances['acc'], test_loss))
        
        
        if val_loss < best_val_loss:
#        if performances['acc'] > best_val_loss:
            torch.save(model,params.best_model_file)
            print('The best model up till now. Saved to File.')
            best_val_loss = val_loss
#            best_val_loss = performances['acc']
        
        torch.cuda.empty_cache()

def test_all_subsystems(model, params):
    model.eval()

    all_feature_names = ['textual','visual','acoustic']
    num_modalities = len(all_feature_names)
    all_subsets = list(powerset(range(num_modalities)))
    for s in all_subsets:
        print(s)
        if len(s)>0 and len(s)<num_modalities:
            test_output,test_target = get_subsystem_predictions(model, params, s, split = 'test')
        
            performances = evaluate(params,test_output,test_target)
            modality_str = '+'.join([all_feature_names[p] for p in s])
            print('model performances for {}:'.format(modality_str))
            performance_str= print_performance(performances, params)
            eval_path = os.path.join('tmp',params.dir_name,'eval_{}'.format(modality_str))
            with open(eval_path,'w') as f:
                f.write(performance_str)
    return performances

def get_subsystem_predictions(model, params, feature_indexes, split ='test'):
    outputs = []
    targets = []
    iterator = None
    if split == 'test':
        iterator = params.reader.get_test(iterable =True, shuffle = False)
    elif split == 'dev':
        iterator = params.reader.get_val(iterable =True, shuffle = False)
    elif split == 'train':
        iterator = params.reader.get_train(iterable =True, shuffle = False)
    else:
        print('Wrong split name. Use test split by default.')
        iterator = params.reader.get_test(iterable =True, shuffle = False)
        
    for _ii,data in enumerate(iterator,0):           
        data_x = [inp.to(params.device) for inp in data[:-1]]
        data_t = data[-1].to(params.device)
#            test_o = model(test_x)          
        
        data_o = model.get_submodality_results(data_x, feature_indexes)
            
        if params.dialogue_format:
            data_t,data_o= flatten_output(data_t,data_o)
        outputs.append(data_o.detach())
        targets.append(data_t.detach())
    
    outputs = torch.cat(outputs)
    targets = torch.cat(targets)
     
    if params.label == 'emotion':
        outputs = outputs.view(targets.shape) 
    return outputs, targets

def get_criterion(params):
    # Only 1-dim output, regression loss is used
    # For monologue sentiment regression
    if params.output_dim ==1:
        criterion = nn.L1Loss()
    # For DialogueRNN and DialogueGCN, MaskedNLLLoss is used
    elif params.dialogue_format and not params.dialogue_context:     
        criterion = MaskedNLLLoss(params.loss_weights)
    # For multi-class classification tasks NLLLoss is used
    else:
        criterion = nn.NLLLoss()
    return criterion

def get_loss(params, criterion, outputs, b_targets, mask):
    # Only 1-dim output, regression loss is used
    # For monologue sentiment regression
    if params.output_dim == 1:
        loss = criterion(outputs,b_targets)
    
    elif params.dialogue_format: 
        b_targets = torch.argmax(b_targets.reshape(-1, params.output_dim),-1)
        outputs = outputs.reshape(-1, params.output_dim)
        if params.dialogue_context:
            loss = criterion(outputs,b_targets)
        else:
            loss = criterion(outputs,b_targets,mask)
            nonzero_idx = mask.view(-1).nonzero()[:,0]
            outputs = outputs[nonzero_idx]
            b_targets = b_targets[nonzero_idx]
            
    # Very rare case of Multi-class classification
    # Treat every class as a binary classification task
    else:
        last_shape = b_targets.shape[-1]
        outputs = outputs.reshape(-1,last_shape)   
        outputs = F.log_softmax(outputs,dim = -1)
        b_targets = torch.argmax(b_targets.reshape(-1, last_shape),-1)
        loss = criterion(outputs,b_targets)
    return b_targets, outputs, loss

def test(model,params):
    model.eval()
    test_output,test_target, test_mask = get_predictions(model, params, split = 'test')    

    if params.dialogue_format: 
        test_target = torch.argmax(test_target.reshape(-1, params.output_dim),-1)
        test_output = test_output.reshape(-1, params.output_dim)
        if not params.dialogue_context:
            nonzero_idx = test_mask.view(-1).nonzero()[:,0]
            test_output = test_output[nonzero_idx]
            test_target = test_target[nonzero_idx]
            
    elif not params.output_dim == 1:
        last_shape = test_target.shape[-1]
        test_output = test_output.reshape(-1,last_shape)   
        test_output = F.log_softmax(test_output,dim = -1)
        test_target = torch.argmax(test_target.reshape(-1, last_shape),-1)
         
    performances = evaluate(params,test_output,test_target)
    if params.network_type == 'cfn':
        test_output_v_l = []
        test_output_l_v = []
        test_output_hard_decision = []
    
        test_target = []
        for _ii,test_data in enumerate(params.reader.get_test(iterable = True, shuffle = False),0):           
            test_x = [inp.to(params.device) for inp in test_data[:-1]]
            test_t = test_data[-1].to(params.device)
            test_o_v_l,test_o_l_v = model.get_decisions(test_x)
            test_hard_decision = model.get_hard_decision(test_x)
            test_output_v_l.append(test_o_v_l)
            test_output_l_v.append(test_o_l_v)
            test_output_hard_decision.append(test_hard_decision)
            test_target.append(test_t)
            
        test_output_v_l = torch.cat(test_output_v_l)
        test_output_l_v = torch.cat(test_output_l_v)
        test_output_hard_decision= torch.cat(test_output_hard_decision)
    
        test_target = torch.cat(test_target)
        
        if params.label == 'emotion':
            test_output_v_l = test_output_v_l.view(test_target.shape)     
            test_output_l_v = test_output_l_v.view(test_target.shape)  
            test_output_hard_decision = test_output_hard_decision.view(test_target.shape)  
     
        performances_v_l = evaluate(params,test_output_v_l,test_target)
        performances_l_v = evaluate(params,test_output_l_v,test_target)
        performances_hard_decision = evaluate(params,test_output_hard_decision,test_target)
        performances = performances_v_l, performances_l_v,performances_hard_decision, performances
    
    return performances

def get_predictions(model, params, split ='dev'):
    outputs = []
    targets = []
    masks = []
    iterator = None
    if split == 'test':
        iterator = params.reader.get_test(iterable =True, shuffle = False)
    elif split == 'dev':
        iterator = params.reader.get_val(iterable =True, shuffle = False)
    elif split == 'train':
        iterator = params.reader.get_train(iterable =True, shuffle = False)
    else:
        print('Wrong split name. Use test split by default.')
        iterator = params.reader.get_test(iterable =True, shuffle = False)
        
    for _ii,data in enumerate(iterator,0):  
        data_x = [inp.to(params.device) for inp in data[:-1]]
        data_t = data[-1].to(params.device)
        data_o = model(data_x)
        
#        if params.dialogue_format:
#            data_t = torch.argmax(data_t.reshape(-1, params.output_dim), -1)
#            data_o = data_o.reshape(-1, params.output_dim)
#            if not params.dialogue_context:
#                mask = data_x[-1]
#                masks.append(mask)
        if params.dialogue_format and not params.dialogue_context:
            masks.append(data_x[-1])
                        
        outputs.append(data_o.detach())
        targets.append(data_t.detach())
            
    outputs = torch.cat(outputs)
    targets = torch.cat(targets)
    if params.dialogue_format and not params.dialogue_context:   
        masks = torch.cat(masks)
        
    return outputs, targets, masks

def save_model(model,params,s):
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    params.dir_name = str(round(time.time()))
    dir_path = os.path.join('tmp',params.dir_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    torch.save(model.state_dict(),os.path.join(dir_path,'model'))
#    copyfile(params.config_file, os.path.join(dir_path,'config.ini'))
    params.export_to_config(os.path.join(dir_path,'config.ini'))
    params_2 = copy.deepcopy(params)
    if 'lookup_table' in params_2.__dict__:
        del params_2.lookup_table
    if 'sentiment_dic' in params_2.__dict__:
        del params_2.sentiment_dic
    del params_2.reader
    pickle.dump(params_2, open(os.path.join(dir_path,'config.pkl'),'wb'))
    
    del params_2
    if 'save_phases' in params.__dict__ and params.save_phases:
        print('Saving Phases.')
        phase_dict = model.get_phases()
        for key in phase_dict:
            file_path = os.path.join(dir_path,'{}_phases.pkl'.format(key))
            pickle.dump(phase_dict[key],open(file_path,'wb'))
    eval_path = os.path.join(dir_path,'eval')
    with open(eval_path,'w') as f:
        f.write(s)
     
def print_performance(performance_dict, params):
    if params.network_type == 'cfn':
        performance_dict_l_v, performance_dict_v_l, performance_dict_hard_decision, overall_performance_dict = performance_dict
        print('l->v performances:')
        _ = _result_print(performance_dict_l_v, params)
        
        print('v->l performances:')
        _ = _result_print(performance_dict_v_l, params)
        
        print('hard_decision performances:')
        _ = _result_print(performance_dict_hard_decision, params)    
    else:
        overall_performance_dict = performance_dict
    print('model performances:')
    performance_str = _result_print(overall_performance_dict, params)
    return performance_str
    
def _result_print(performance_dict, params):
    performance_str = ''
    if params.label == 'sentiment' or params.dialogue_format:
        for key, value in performance_dict.items():
            performance_str = performance_str+ '{} = {} '.format(key,value)
    elif params.label == 'emotion':
        performance_str = performance_str +'acc = {}\n'.format(performance_dict['acc'])
        if params.dataset_name == 'iemocap':
            emotions = ["Neutral", "Happy", "Sad", "Angry"]
            acc_per_class = performance_dict['acc_per_class']
            f1_per_class = performance_dict['f1_per_class']
            for i in range(4):
                performance_str = performance_str + '{}: acc = {}, f1 = {}\n'.format(emotions[i],acc_per_class[i],f1_per_class[i])
    print(performance_str)
    return performance_str

def case_study_subsystem(params, model, feature_indexes):
    data_file = os.path.join(params.pickle_dir_path, params.dataset_name+'_data.pkl')
    dataset = pickle.load(open(data_file,'rb'))
    
    all_feature_names = ['textual','visual','acoustic']
    modality_str = '+'.join([all_feature_names[p] for p in feature_indexes])
 
    if params.true_labels:
        output_file = os.path.join('labels_per_video','true_labels_'+modality_str+'_'+params.dataset_name+'.txt')
        true_labels_writer = open(output_file, 'w')
        
    if params.model_prediction:
         output_file = os.path.join('labels_per_video',modality_str+'_'+params.network_type+'_'+params.dataset_name+'.txt')
         model_prediction_writer = open(output_file, 'w')
         
    if params.per_sample_analysis:
        output_file = os.path.join('labels_per_video','per_sample_analysis_'+modality_str+'_'+params.network_type+'_'+params.dataset_name+'.txt')
        per_sample_analysis_writer = open(output_file, 'w')

    for data in [dataset['train'],dataset['test'],
                  dataset['valid']]:
        true_labels = data['labels']
        texts = data['text']
        ids = data['ids']
        intervals = data['intervals']
        model.eval()
        
        dic = params.reader.embedding.dictionary
        dic = list(dic.keys())
        
        data['text'] = [params.reader.embedding.text_to_sequence(x) for x in data['text']]
                    
        X = [data['text'],data['vision'],data['audio']]
        X = [torch.tensor(x,dtype = torch.int64) if i==0 else torch.tensor(x,dtype = torch.float32) for i,x in enumerate(X)]  

        X = [s.to(params.device) for s in X]
        
        # Always include textual modality 
        feature_indexes = params.feature_indexes
        if 0 not in feature_indexes:
            feature_indexes = [0]+feature_indexes
        X = [_x for i,_x in enumerate(X) if i in feature_indexes]
        num_samples = len(X[0])
        num_batches = int(np.ceil(num_samples/params.batch_size))
        outputs = []
        for i in range(num_batches):
            batch_x = [_x[i*params.batch_size:min((i+1)*params.batch_size,num_samples)] for _x in X]
            batch_output = model.get_submodality_results(tuple(batch_x),feature_indexes)             
            if params.use_sentiment_dic:
                outputs.append(batch_output[0].detach())
            else:
                outputs.append(batch_output.detach())
            torch.cuda.empty_cache()
#            GPUtil.showUtilization()
            
        outputs = torch.cat(outputs)
        
        for _id, _interval, _text, _output, _label in zip(ids, intervals, texts, outputs, true_labels):
            text_sequence = " ".join([_w for _w in _text if not _w == 'UNK'])
            if params.true_labels:
                true_labels_writer.write('{}\t[{:.3f},{:.3f}]\t{}\t{:.2f}\n'.format(_id, _interval[0], _interval[1], text_sequence,_label[0][0]))
            
            if params.model_prediction:
                model_prediction_writer.write('{}\t[{:.3f},{:.3f}]\t{}\t{:.2f}\n'.format(_id.split('[')[0], _interval[0], _interval[1], text_sequence,_output[0].cpu().numpy()))
            
            if params.per_sample_analysis:
                per_sample_analysis_writer.write('{}\t{:.2f}\t{:.2f}\n'.format(text_sequence,_output[0].cpu().numpy(),_label[0][0]))
    if params.true_labels:
        true_labels_writer.close()
    if params.model_prediction:
        model_prediction_writer.close()
    if params.per_sample_analysis:
        per_sample_analysis_writer.close()
        
def case_study(params, model):
    data_file = os.path.join(params.pickle_dir_path, params.dataset_name+'_data.pkl')
    dataset = pickle.load(open(data_file,'rb'))
    
    # IEMOCAP Dataset has no ids available, so only per_sample_analysis can be applied here
    if params.label == 'emotion' and params.dataset_name == 'iemocap': 
        output_file = os.path.join('labels_per_video','per_sample_analysis_'+params.network_type+'_'+params.dataset_name+'.txt')
        per_sample_analysis_writer = open(output_file, 'w')
        for data in [dataset['train'],dataset['test'],
                  dataset['valid']]:
            true_labels = data['labels']
            texts = data['text']
            model.eval()
        
            dic = params.reader.embedding.dictionary
            dic = list(dic.keys())
        
            data['text'] = [params.reader.embedding.text_to_sequence(x) for x in data['text']]
        
            X = [data['text'],data['vision'],data['audio']]
            X = [torch.tensor(x,dtype = torch.int64) if i==0 else torch.tensor(x,dtype = torch.float32) for i,x in enumerate(X)]  

            X = [s.to(params.device) for s in X]
            # Always include textual modality 
            feature_indexes = params.feature_indexes
            if 0 not in feature_indexes:
                feature_indexes = [0]+feature_indexes
            X = [_x for i,_x in enumerate(X) if i in feature_indexes]
            num_samples = len(X[0])
            num_batches = int(np.ceil(num_samples/params.batch_size))
            outputs = []
            for i in range(num_batches):
                batch_x = [_x[i*params.batch_size:min((i+1)*params.batch_size,num_samples)] for _x in X]
                batch_output = model(tuple(batch_x)) 
                if params.use_sentiment_dic:
                    outputs.append(batch_output[0].detach())
                else:
                    outputs.append(batch_output.detach())
                torch.cuda.empty_cache()
#            GPUtil.showUtilization()
            
            outputs = torch.cat(outputs)
#            outputs = model(tuple(X))
            
            for _text, _output, _label in zip(texts, outputs, true_labels):
                text_sequence = " ".join([_w for _w in _text if not _w == 'UNK'])
                per_sample_analysis_writer.write('{}\t{:.2f}\t{:.2f}\n'.format(text_sequence,_output[0].cpu().numpy(),_label[0][0]))
        per_sample_analysis_writer.close()
        return
 
    if params.true_labels:
        output_file = os.path.join('labels_per_video','true_labels_'+params.dataset_name+'.txt')
        true_labels_writer = open(output_file, 'w')
        
    if params.model_prediction:
         output_file = os.path.join('labels_per_video',params.network_type+'_'+params.dataset_name+'.txt')
         model_prediction_writer = open(output_file, 'w')
         
    if params.per_sample_analysis:
        output_file = os.path.join('labels_per_video','per_sample_analysis_'+params.network_type+'_'+params.dataset_name+'.txt')
        per_sample_analysis_writer = open(output_file, 'w')

        
    for data in [dataset['train'],dataset['test'],
                  dataset['valid']]:
        true_labels = data['labels']
        texts = data['text']
        ids = data['ids']
        intervals = data['intervals']
        model.eval()
        
        dic = params.reader.embedding.dictionary
        dic = list(dic.keys())
        
        data['text'] = [params.reader.embedding.text_to_sequence(x) for x in data['text']]
                    
        X = [data['text'],data['vision'],data['audio']]
        X = [torch.tensor(x,dtype = torch.int64) if i==0 else torch.tensor(x,dtype = torch.float32) for i,x in enumerate(X)]  

        X = [s.to(params.device) for s in X]
        
        # Always include textual modality 
        feature_indexes = params.feature_indexes
        if 0 not in feature_indexes:
            feature_indexes = [0]+feature_indexes
        X = [_x for i,_x in enumerate(X) if i in feature_indexes]
        num_samples = len(X[0])
        num_batches = int(np.ceil(num_samples/params.batch_size))
        outputs = []
        for i in range(num_batches):
            batch_x = [_x[i*params.batch_size:min((i+1)*params.batch_size,num_samples)] for _x in X]
            batch_output = model(tuple(batch_x))             
            if params.use_sentiment_dic:
                outputs.append(batch_output[0].detach())
            else:
                outputs.append(batch_output.detach())
            torch.cuda.empty_cache()
#            GPUtil.showUtilization()
            
        outputs = torch.cat(outputs)
        
        for _id, _interval, _text, _output, _label in zip(ids, intervals, texts, outputs, true_labels):
            text_sequence = " ".join([_w for _w in _text if not _w == 'UNK'])
            if params.true_labels:
                true_labels_writer.write('{}\t[{:.3f},{:.3f}]\t{}\t{:.2f}\n'.format(_id, _interval[0], _interval[1], text_sequence,_label[0][0]))
            
            if params.model_prediction:
                model_prediction_writer.write('{}\t[{:.3f},{:.3f}]\t{}\t{:.2f}\n'.format(_id.split('[')[0], _interval[0], _interval[1], text_sequence,_output[0].cpu().numpy()))
            
            if params.per_sample_analysis:
                per_sample_analysis_writer.write('{}\t{:.2f}\t{:.2f}\n'.format(text_sequence,_output[0].cpu().numpy(),_label[0][0]))
    if params.true_labels:
        true_labels_writer.close()
    if params.model_prediction:
        model_prediction_writer.close()
    if params.per_sample_analysis:
        per_sample_analysis_writer.close()

def print_result_from_dir(dir_path):
    params = Params()
    params.parse_config(os.path.join(dir_path, 'config.ini'))
    reader = open(os.path.join(dir_path,'eval'),'r')
    s = reader.readline().split()
    print('dataset: {}, network_type: {}, acc: {}, f1:{}'.format(params.dataset_name,params.network_type,s[2],s[5]))
    
def save_performance(params, performance_dict):
    if params.network_type == 'cfn':
        output_file = 'eval/{}_{}_invidual_decisions.csv'.format(params.dataset_name, params.network_type)
        performance_dict_l_v, performance_dict_v_l, performance_dict_hard_decision, performance_dict = performance_dict
        df = pd.DataFrame()
        output_dic = {'dataset' : params.dataset_name,
                        'modality' : params.features,
                        'network' : params.network_type,
                        'decision':'l->v'}
        
        output_dic.update(performance_dict_l_v)
        df = df.append(output_dic, ignore_index = True)
        output_dic = {'dataset' : params.dataset_name,
                        'modality' : params.features,
                        'network' : params.network_type,
                        'decision':'v->l'}
        output_dic.update(performance_dict_v_l)
        df = df.append(output_dic, ignore_index = True)
        
        output_dic = {'dataset' : params.dataset_name,
                        'modality' : params.features,
                        'network' : params.network_type,
                        'decision':'hard'}
        output_dic.update(performance_dict_hard_decision)
        df = df.append(output_dic, ignore_index = True)
        df.to_csv(output_file, encoding='utf-8', index=True)
        
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
