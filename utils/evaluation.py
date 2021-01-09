# -*- coding: utf-8 -*-

import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def evaluate(params, outputs, targets):
    if params.label == 'sentiment':
        # Single real-valued output
        
        # Binary ACC
        n_correct = ((outputs>=0) == (targets>=0)).sum().item()
        n_total = len(outputs)
        acc_2 = n_correct / n_total
        
        # Binary F1
        outputs_np = outputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        f1 = f1_score((targets_np>=0),(outputs_np>=0),average='weighted')
            
        # Correlation
        corr = np.corrcoef(outputs_np.transpose(), targets_np.transpose())[0][1]    
        
        # MAE
        mae = torch.mean(torch.abs(outputs-targets)).item()
        
        # Accuracy for multiclass
        n_correct = sum(np.round(targets_np)==np.round(outputs_np))[0]
        acc_7 = n_correct/n_total
        
        targets_clamped = np.clip(targets_np, a_min = -2, a_max = 2)
        outputs_clamped = np.clip(outputs_np, a_min = -2, a_max = 2)
        
        n_correct = sum(np.round(targets_clamped)==np.round(outputs_clamped))[0]
        acc_5 = n_correct/n_total

        performance_dict = {'acc':acc_2,'binary_f1':f1,'accuracy_5':acc_5,'accuracy_7':acc_7,'MAE':mae,'r':corr }
        
    else:
#        emos = ["Neutral", "Happy", "Sad", "Angry"]
        # outputs is of shape (batch_size, num_classes, 2)
        # targets is of shape (batch_size, num_classes, 2)
        outputs_max_ids = outputs.argmax(dim = -1).t()
        targets_max_ids = targets.argmax(dim = -1).t()

        num_classes,n_total = outputs_max_ids.shape        
        f1_per_class = []
        acc_per_class = []
        for class_i in range(num_classes):

            output_classes = outputs_max_ids[class_i].cpu().numpy()
            target_classes = targets_max_ids[class_i].cpu().numpy()
            f1 = f1_score(target_classes, output_classes, average='weighted')
            acc = accuracy_score(target_classes, output_classes)
            acc_per_class.append(acc)
            f1_per_class.append(f1)
        
        acc = float(torch.sum(outputs_max_ids == targets_max_ids))/float(num_classes*n_total)
        performance_dict = {'acc':acc,'f1_per_class':f1_per_class, 'acc_per_class':acc_per_class}

    return performance_dict