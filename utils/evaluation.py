# -*- coding: utf-8 -*-

import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def evaluate(params, outputs, targets):
    if params.dialogue_format:
        outputs_max_ids = outputs.argmax(dim = -1).cpu().numpy()
        targets_max_ids = targets.cpu().numpy()
        acc = accuracy_score(outputs_max_ids, targets_max_ids)
        weighted_f1 = f1_score(targets_max_ids,outputs_max_ids,average='weighted')
        performance_dict = {'acc':acc,'f1':weighted_f1}
        
        # To do:
        # Accuracy per class= |pos_pred and pos_output|/|pos_pred or pos_output|
        report = classification_report(targets_max_ids, outputs_max_ids, target_names=params.emotion_dic, output_dict = True)
    

        for _id, emo in enumerate(params.emotion_dic):
            intersection = sum((outputs_max_ids == _id) * (targets_max_ids == _id))
            union = sum(((outputs_max_ids == _id) + (targets_max_ids == _id))>0)
            acc = intersection/union
            f1 = report[emo]['f1-score'] 
           
            
            #WA = (TP *N/P + TN)/2N
            performance_dict[emo+'_f1'] = f1
            performance_dict[emo+'_precision'] = report[emo]['precision']
            performance_dict[emo+'_recall'] = report[emo]['recall']

            
    else:    
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
            # outputs and targets are of shape (batch_size, num_classes, 2)
    
            outputs_max_ids = outputs.argmax(dim = -1).t()
            targets_max_ids = targets.argmax(dim = -1).t()
            
            num_classes,n_total = outputs_max_ids.shape
            
            f1_per_class = []
            acc_per_class = []
            for class_i in range(num_classes):
    #            print("{}: ".format(emos[class_i]))
    
                output_classes = outputs_max_ids[class_i].cpu().numpy()
                target_classes = targets_max_ids[class_i].cpu().numpy()
                f1 = f1_score(target_classes, output_classes, average='weighted')
                acc = accuracy_score(target_classes, output_classes)
                acc_per_class.append(acc)
                f1_per_class.append(f1)
    
    #            print("  - F1 Score: ", f1)
    #            print("  - Accuracy: ", acc)
            
            acc = float(torch.sum(outputs_max_ids == targets_max_ids))/float(num_classes*n_total)
            performance_dict = {'acc':acc,'f1_per_class':f1_per_class, 'acc_per_class':acc_per_class}
    
    #        last_shape = outputs.shape[-1]
    #        num_classes = outp-2]
    #        num_classes = len(outputs[0])
    #        
    #        # Convert outputs to one-hotuts.shape[ format
    #        one_hot_outputs = torch.zeros_like(outputs)
    #        max_ids = outputs.argmax(dim = -1)
    #        for i in range(n_total):
    #            for j in range(len(max_ids[i])):
    #                one_hot_outputs[i, j,max_ids[i,j]] = 1
    #        
    #        true_positives = torch.sum(one_hot_outputs * targets, dim = 0)
    #        false_positives = torch.sum(torch.clamp(one_hot_outputs - targets, 0, 1), dim=0)
    #        false_negatives = torch.sum(torch.clamp(targets-one_hot_outputs , 0, 1), dim=0)
    #        true_negatives = torch.sum((1-one_hot_outputs) *(1-targets), dim = 0)
    #
    #        
    ##        print("True Positives per class : ", true_positives)
    ##        print("False Positives per class : ", false_positives)
    ##        print("False Negatives per class : ", false_negatives)
    #        
    #        # ------------- Macro level calculation ---------------
    #        macro_precision = 0
    #        macro_recall = 0
    #        
    #        f1_per_class = []
    #        acc_per_class = []
    #        for i in range(num_classes):
    #            precision = true_positives[i] / (true_positives[i] + false_positives[i]) if (true_positives[i] + false_positives[i]) >0 else 0
    #            macro_precision += precision
    #            recall = true_positives[i] / (true_positives[i] + false_negatives[i]) if (true_positives[i] + false_negatives[i]) >0 else 0
    #            accuracy = (true_negatives[i] + true_positives[i])/(true_negatives[i] + true_positives[i] + false_negatives[i]+ false_positives[i])
    #            macro_recall += recall
    #            f1 = ( 2 * recall * precision ) / (precision + recall) if (precision+recall) > 0 else 0
    #            f1_per_class.append(float(f1))
    #            acc_per_class.append(float(accuracy))
    ##            print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (i, precision, recall, f1))
    #    
    #        macro_precision /= num_classes
    #        macro_recall /= num_classes
    ##        macro_f1 = (2 * macro_recall * macro_precision ) / (macro_precision + macro_recall) if (macro_precision+macro_recall) > 0 else 0
    ##        print("Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macro_precision, macro_recall, macro_f1))
    #    
    #        # ------------- Micro level calculation ---------------
    #        true_positives = true_positives.sum()
    #        false_positives = false_positives.sum()
    #        false_negatives = false_negatives.sum()
    #    
    ##        print("Micro TP : %d, FP : %d, FN : %d" % (true_positives, false_positives, false_negatives))
    #    
    #        micro_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    #        micro_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    #    
    #        micro_f1 = ( 2 * micro_recall * micro_precision ) / (micro_precision + micro_recall) if (micro_precision+micro_recall) > 0 else 0
    #        # -----------------------------------------------------
    #
    #        accuracy = float(torch.sum(outputs.argmax(dim = 1)==targets.argmax(dim = 1)))/float(n_total)
    #
    ##        print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, micro_precision, micro_recall, micro_f1))       
    #        performance_dict = {'acc':accuracy,'micro_f1':micro_f1,'f1_per_class':f1_per_class}
    
            # The WA and F1 are computed on a binary basis         
            #Emotion Classification, output Weighted Average (WA) and F1
            #WA = (TP *N/P + TN)/2N

    return performance_dict