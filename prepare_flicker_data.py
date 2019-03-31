import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from scipy import stats
import re

def load_data(path):
    all_files = os.listdir(path)
    txt_list = []
    img_list = []
    for _file in all_files:
        txt = open(os.path.join(path, _file)).readline()
        txt_list.append(txt)
        img = _file.split(".")
        img = img[0]
        img_list.append(img)

    return np.array(txt_list),np.array(img_list) 
 


txt_list_neg_test, img_list_neg_test = load_data("datasets/Flickr/data/flickr-dataset/documents/testingset/negative")
y_list_neg = np.array([-1]*len(txt_list_neg_test))

txt_list_pos_test, img_list_pos_test = load_data("datasets/Flickr/data/flickr-dataset/documents/testingset/positive")
y_list_pos = np.array([1]*len(txt_list_pos_test))


columns = ['txt', 'img', 'y']
df_ = pd.DataFrame(columns=columns)


df_['txt'] = np.concatenate((txt_list_neg_test,txt_list_pos_test), axis=0)
df_['img'] = np.concatenate((img_list_neg_test,img_list_pos_test), axis=0)
df_['y'] = np.concatenate((y_list_neg,y_list_pos), axis=0)



#Count the number of words in text
lengths = []
for index, row in df_.iterrows():
    words = str(row['txt']).split(' ')
    length = len(words)
    lengths.append(length)
    
df_['txt_lenth'] =  lengths 


#Cound the positive words
pos_words = []

with open('data/positive_words.txt', 'r') as f:
    lines = f.readlines()
    pos_words = [line.strip('\n') for line in lines]
   

count_pos_words = []
for index, row in df_.iterrows():
    total = 0
    total = len([word for word in str(row['txt']).split(' ') if word in pos_words])
    count_pos_words.append(total)
    
df_['pos_words'] =  count_pos_words  


#Cound the negative words
neg_words = []

with open('data/negative_words.txt', 'r') as f:
    lines = f.readlines()
    neg_words = [line.strip('\n') for line in lines]
   

count_neg_words = []
for index, row in df_.iterrows():
    total = 0
    total = len([word for word in str(row['txt']).split(' ') if word in neg_words])
    count_neg_words.append(total)
    
df_['neg_words'] =  count_neg_words


df_.to_csv('data/test.csv', encoding='utf-8', index=True)

#Analyse data

print('Test dataset is done!')
exit
print('Exit')


  
    

#print('Positive:', len(df.loc[df['y'] == 1])) 
#print('Negative:', len(df.loc[df['y'] == -1]))   


#pos = df.loc[df['y'] == 1]
#neg = df.loc[df['y'] == -1]

#print('Average Length:', pos['txt_lenth'].mean())
#print('Max Length:', pos['txt_lenth'].max())   
#print('Min Length:', pos['txt_lenth'].min())


#print('Average:',  neg['txt_lenth'].mean())
#print('Max:', neg['txt_lenth'].max())   
#print('Min:', neg['txt_lenth'].min())



#t, p = ttest_ind(pos['txt_lenth'], neg['txt_lenth'], equal_var=False)
#print("ttest_ind: t = %g  p = %g" % (t, p))

#print('Average Positive:', pos['pos_words'].mean())
#print('Average Positive:', pos['neg_words'].mean())

#print('Average Positive:', neg['pos_words'].mean())
#print('Average Positive:', neg['neg_words'].mean())

#t, p = ttest_ind(pos['pos_words'], neg['neg_words'], equal_var=False)
#print("ttest_ind: t = %g  p = %g" % (t, p))

#ambiguous = (df["pos_words"] == 1) & (df["neg_words"] == 1)
#total = len([x for x in ambiguous if x is True])


#df = df.drop('Unnamed: 0', axis=1)
#df.to_csv('data/train.csv', encoding='utf-8', index=True)


#Just to check if the models performs better.

#import numpy as np

#a = np.zeros(19861, dtype = int)
#b = np.zeros(19861, dtype = int)


#for i in range (0,len(a)):
#    if (i<11857):
#        a[i] = 1
        
#for i in range (0,len(a)):
#    if (i<18013):
 #       a[i] = 1  
        
#t, p = ttest_ind(a.tolist(), b.tolist(), equal_var=True)
#print("ttest_ind: t = %g  p = %g" % (t, p))        