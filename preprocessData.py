import numpy as np
import pandas as pd 
import re
import pickle

from nltk.corpus import stopwords
import string
import operator
import collections as c
from functions.load_files import *
from keras.preprocessing.text import Tokenizer, one_hot
from sklearn.model_selection import train_test_split


class preprocessData():
     
        def  __init__(self, data_path, lower, alphabetic, punctuation,whitespaces,removestopwords,_ascii):
             self.data_path =  data_path
             self.lower = lower
             self.alphabetic = alphabetic
             self.punctuation = punctuation
             self.whitespaces = whitespaces
             self.removestopwords = removestopwords
             self._ascii = _ascii
            
             
        def loadText(self):      
            dataset = pd.read_csv(self.data_path, sep=',') #, names = ["Index", "Img", "Question", "Answer"])
            dataset = dataset.replace(np.nan, '', regex=True)
            
            imgs = dataset.iloc[:,2].tolist()
            texts = dataset.iloc[:,1].tolist()
        
            
            return imgs,texts
        
        def textVocab(docs, ovv_token=False):
            
            if  ovv_token == True: 
                tokenizer = Tokenizer(oov_token='UNK')
            else:    
                tokenizer = Tokenizer()
            
            #tokenizer = Tokenizer()
            tokens = tokenizer.fit_on_texts(docs)
            
            
            if  ovv_token == True:
                vocab_size = len(tokenizer.word_index) + 1
               
            else:    
                vocab_size = len(tokenizer.word_index) 
            
            word_index = tokenizer.word_index
            
            
            
            
            print('Vocabulary size: %s' % vocab_size)
            
            # integer encode the documents
            #encoded_docs = tokenizer.texts_to_sequences(docs)
            
            return tokenizer
        
        def textProcess(self, docs):
            
            texts = []
            
            for doc in docs:
                
                
                # split into tokens by white space
                doc = str(doc)
                
                 #Remove ascii characters
                if  self._ascii == True: 
                   doc = re.sub(r'[^\x00-\x7F]+',' ', doc)
    
                tokens = doc.split()
                
                #Convert text to lowercase
                if  self.lower == True: 
                    tokens = [word.lower() if word!='UKN' else 'UKN' for word in tokens]
                    
                #Remove punctuation
                if  self.punctuation == True: 
                    tokens = [re.sub(r'[^\w\s]','',word) for word in tokens]
                    
                    
                # Remove white spaces
                if  self.whitespaces == True: 
                    tokens = [word.strip() for word in tokens ]  
                    
              
                    
                    
                    
                # Remove remaining tokens that are not alphabetic
                if  self.alphabetic == True: 
                    tokens = [word for word in tokens if word.isalpha()]
                    
                # filter out stop words 
                if  self.removestopwords == True:
                    stop_words = set(stopwords.words('english'))
                    #We want to remove the word No,Yes from the list.
                    remove = ['no', 'yes']
                    new_stop_words = [word for word in stop_words if word not in remove]
                    tokens = [w for w in tokens if not w in new_stop_words]
                    
                    
                # filter out short tokens
                #tokens = [word for word in tokens if len(word) > 1]
	            
                texts.append(" ".join(str(w) for w in tokens))
                    
                
            return texts
        
        
        
        
def getUniqueWords(allWords) :
            uniqueWords = [] 
            for i in allWords:
                if not i in uniqueWords:
                    uniqueWords.append(i)
            return uniqueWords
        
 
         
        
def main ():
    
    #Load data and crete test and train data set
    data = pd.read_csv('data/data.csv', sep=',')
    data = data.drop('Unnamed: 0', axis=1)
    data = data.replace(np.nan, '', regex=True)
    
    #Cound the positive words
    #en_words = []
    
    #with open('data/enlglist_words.txt', 'r') as f:
        #lines = f.readlines()
        #en_words = [line.strip('\n') for line in lines]
        
    #en_words = [en_word.lower() for en_word in en_words]

    #Replance non english words with UNK words
    #en_texts = [] 
    #replaceToken='UNK'
    #for text in data['txt']:
    #    text = ' '.join([word.lower() if word.lower() in en_words else replaceToken for word in text.split()])
    #    en_texts.append(text)
        
    #data['txt'] = en_texts 
    
    train, test = train_test_split(data, test_size=0.2)
    train = train.reset_index(drop=True)
    #train['IDs'] = train.index.get_values()
    test = test.reset_index(drop=True)
    #test['IDs'] = test.index.get_values()

    train.to_csv('data/train.csv', encoding='utf-8', index=True)
    test.to_csv('data/valid.csv', encoding='utf-8', index=True)    
        
    
    
    
    # Parameters
    params = {'data_path': 'data/train.csv',
              'lower': True,
              'alphabetic': False,
              'punctuation': True,
              'whitespaces': True,
              'removestopwords': False,
              '_ascii': True}
    
    #Initialize the data
    p = preprocessData(**params)
    
    #Retrieve imgs and texts
    imgs,texts = p.loadText()
    
    
    #Pre-processing     
    pre_texts = p.textProcess(texts)
    
    
    #Replace tokens which appearing less than 4 times with unknown word.
    #word_counter = c.Counter(" ".join(texts).split()).most_common(5580) 
    #top_words = []
    #for w in word_counter:
    #    top_words.append(w[0])
        
    #topwords_texts = [] 
    #replaceToken='UNK'
    #for text in texts:
    #    text = ' '.join([word if word in top_words else replaceToken for word in text.split()])
    #    topwords_texts.append(text)
    

    
   

    #Tokenizer    
    tokenizer = preprocessData.textVocab(pre_texts, True)  
    word_index = tokenizer.word_index
    count_thres = 3
    low_count_words = [w for w,c in tokenizer.word_counts.items() if c < count_thres]
    for w in low_count_words:
        del tokenizer.word_index[w]
        del tokenizer.word_docs[w]
        del tokenizer.word_counts[w]
   
    
    #sorted_word_index = sorted(word_index.items(), key=operator.itemgetter(1))
    
    
    save_to_pickle('data/tokenizer.pickle', tokenizer)
    
    
    
       
    

   
 
    
            
          

if __name__ == '__main__':
    main()        
    
    

        
        