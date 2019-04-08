#VGG19 + multichanel CNN
from keras.applications import VGG16, VGG19, InceptionV3, Xception
from keras.models import Model
from keras.layers import Lambda,Multiply,Concatenate, Input,Dense, Flatten, Dropout, Conv1D,GlobalAveragePooling2D, Conv2D,LSTM,Embedding, MaxPooling1D, MaxPooling2D, ZeroPadding2D, concatenate
from models.BasicModel import BasicModel
import keras.backend as K

import pandas as pd#
import pickle


class uni_text(BasicModel):
    def initialize(self):
        
        #text
        self.max_seq_length = self.opt.max_seq_length
        self.vocab_size = self.opt.vocab_size
        self.embedding_dim = self.opt.embedding_dim
        self.embedding_matrix = self.opt.embedding_matrix
        
        
        
    def  __init__(self, opt):
        #super(lstm_normI, self).__init__(opt)
        super().__init__(opt)
        
   
    def text_model(self):
        #We will define a model with three input channels for processing 
        # 1-grams, 2-grams, and 3-grams .
        txt_input = Input(shape=(self.max_seq_length,), name='txt_input')
       
        # channel 1
        embedding1  = Embedding(input_dim=self.vocab_size, 
                                      output_dim=self.embedding_dim, 
                                      weights=[self.embedding_matrix],
                                      input_length=self.max_seq_length)(txt_input)
        conv1  = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
        drop1  = Dropout(0.5)(conv1)
        pool1  = MaxPooling1D(pool_size=2)(drop1)
        flat1  = Flatten()(pool1)
        
        # channel 2
        embedding2 = Embedding(input_dim=self.vocab_size, 
                                      output_dim=self.embedding_dim, 
                                      weights=[self.embedding_matrix],
                                      input_length=self.max_seq_length)(txt_input)
        conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
        drop2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat2 = Flatten()(pool2)
        
        # channel 3
        embedding3 = Embedding(input_dim=self.vocab_size, 
                                      output_dim=self.embedding_dim, 
                                      weights=[self.embedding_matrix],
                                      input_length=self.max_seq_length)(txt_input)
        conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = Flatten()(pool3)
        
        # merge
        merged = Concatenate()([flat1, flat2, flat3])
        #merged  = Concatenate([, , flat3])
        
        text = Dense(512, activation='tanh' , name='txt_output')(merged)
        
        
        return Model(inputs=txt_input, outputs=text)
    
    
        
    
    
    
        
    def build(self):    
        
         print('Multi-channel CNN model...')
        
         txt_model = self.text_model()
         
         
         mergedOut = txt_model.get_layer('txt_output').output
        
          
         mergedOut = Dense(2, activation='softmax' )(mergedOut) 
         
         
         model = Model(txt_model.get_layer('txt_input').output, mergedOut)
         
         
         
         return model