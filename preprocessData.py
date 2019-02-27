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



class preprocessData():
     
        def  __init__(self, data_path, lower, alphabetic, punctuation,whitespaces,removestopwords):
             self.data_path =  data_path
             self.lower = lower
             self.alphabetic = alphabetic
             self.punctuation = punctuation
             self.whitespaces = whitespaces
             self.removestopwords = removestopwords
            
             
        def loadText(self):      
            dataset = pd.read_csv(self.data_path, sep=',') #, names = ["Index", "Img", "Question", "Answer"])
            
            imgs = dataset.iloc[:,2].tolist()
            questions = dataset.iloc[:,3].tolist()
            answers = dataset.iloc[:,1].tolist()
            
            return imgs,questions,answers
        
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
            
            return tokenizer, word_index, vocab_size
        
        def textProcess(self, docs):
            
            texts = []
            
            for doc in docs:
                
                
                # split into tokens by white space
                tokens = doc.split()
                
                #Convert text to lowercase
                if  self.lower == True: 
                    tokens = [word.lower() for word in tokens]
                    
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
    # Parameters
    params = {'data_path': 'data/DAQUAR_Train.csv',
              'lower': True,
              'alphabetic': False,
              'punctuation': True,
              'whitespaces': True,
              'removestopwords': False}
    
    #Initialize the data
    p = preprocessData(**params)
    
    
    #Retrieve imgs, questions, and answers
    imgs,questions,answers = p.loadText()
    
    #Pre-processing     
    questions = p.textProcess(questions)
    answers = p.textProcess(answers)
    
    
    
    ################ Preprocess Questions ################
             
    #Get the most 1000 frequent words in questions
    #top1000_questions = c.Counter(" ".join(questions).split()).most_common(999)
    #top_words_questions = []
    #for w in top1000_questions:
    #    top_words_questions.append(w[0])    
        
   
    #additional special UKN token will be used to map infrequent words
    #processed_questions = []
    #for question in questions:
    #    tokens = [w  if  w in top_words_questions else 'UNK' for w in question.split()]
    #    processed_questions.append(" ".join(str(w) for w in tokens))     
        
        
        
    tokenizer, word_index, vocab_size = preprocessData.textVocab(questions, ovv_token=True)     
        
    #sorted_word_index = sorted(word_index.items(), key=operator.itemgetter(1))
    
    
    save_to_pickle('data/word_index.pickle', word_index)
    save_to_pickle('data/tokenizer.pickle', tokenizer)
         
    

    ############ END Preprocess Questions ################
    
    
    
    ################ Preprocess Answers ################
    
    #Get the most 1000 frequent words in answers
    #top1000_answers = c.Counter(" ".join(answers).split()).most_common(1000)
    #top_words_answers = []
    #for w in top1000_answers:
    #    top_words_answers.append(w[0])
        
        
   #Short words   
   
        
        
   #save_to_pickle('data/top_words_answers.pickle', top_words_answers)    
        
      
        
        
    #map words to the most common words
    #processed_answers = []
    #for answer in answers:
    #    tokens = [w  for w in answer.split() if  w in top_words_answers ]
    #    processed_answers.append(" ".join(str(w) for w in tokens))     
   
    
   
    
    #Save all classes
    categories = getUniqueWords(answers)
    
     
    
    #tokenizer, word_index, vocab_size = preprocessData.textVocab(answers) 
    
    #save_to_pickle('data/answer_tokenizer.pickle', tokenizer)
     
    
    #encoded_docs = tokenizer.texts_to_sequences(answers)
    
    
    cat_words = []
    for i,w in enumerate(categories):
        cat_words.append(w)
   
    save_to_pickle('data/category_words.pickle', cat_words)    
   
    df_ = pd.DataFrame(columns=cat_words)
    for r, answer in enumerate(answers):
        row = []
        for category in categories:
            if category == answer:
                encode = 1
            else:
                encode = 0
            row.append(encode)
        df_.loc[r] = row    
    
    #for r, encoded_doc in enumerate(encoded_docs):
    #    row = []
    #    for c, w in enumerate(vocab_words):
    #        if c+1 in encoded_doc: #the index exist in the encoded doc.
    #            cat = 1
    #        else:
    #            cat = 0
    #        row.append(cat)    
    #    df_.loc[r] = row 
        
    df_['questions'] = questions    
    
    df_['imgs'] = imgs 
        
    
    #export datafrane to csv
    df_.to_csv('data/train.csv', encoding='utf-8', index=True)
    
    
    ################ Preprocess Answers ################
    
    
    
    ################ Valid Data ################
    
    #Vocabulary builded on questions only
    
    #top_words_answers = load_pickle('data/top_words_answers.pickle')
    
    #tokenizer = load_pickle('data/answer_tokenizer.pickle')
    
    categories = load_pickle('data/category_words.pickle')
    
    
    
    
    
    #Map words to top_words_answers[]
    #processed_answers = []
    #for answer in answers:
    #    tokens = [w  for w in answer.split() if  w in top_words_answers ]
    #    processed_answers.append(" ".join(str(w) for w in tokens)) 
        
    
    # integer encode the documents
    #encoded_valid_docs = tokenizer.texts_to_sequences(answers)
    
    cat_words = []
    for i,w in enumerate(categories):
        cat_words.append(w)
        
    df_valid = pd.DataFrame(columns=cat_words)
    for r, answer in enumerate(answers):
        row = []
        for category in categories:
            if category == answer:
                encode = 1
            else:
                encode = 0
            row.append(encode)
        df_valid.loc[r] = row  
        
    df_valid['questions'] = questions    
    df_valid['imgs'] = imgs 
    
    
     #export datafrane to csv
    df_valid.to_csv('data/valid.csv', encoding='utf-8', index=True)
    
            
          

if __name__ == '__main__':
    main()        
    
    

        
        