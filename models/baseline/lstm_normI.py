#Deeper LSTM Q + norm I

from keras.models import Model
from keras.layers import Lambda,Multiply, Input,Dense, Flatten, Dropout, Conv2D,LSTM,Embedding, MaxPooling2D, ZeroPadding2D, Concatenate
from models.BasicModel import BasicModel
import keras.backend as K

import pandas as pd#
import pickle

class lstm_normI(BasicModel):
    def initialize(self):
         print('Initialize...')
         self.max_seq_length = self.opt.max_seq_length
         self.vocab_size = self.opt.vocab_size
         self.embedding_dim = self.opt.embedding_dim
         self.embedding_matrix = self.opt.embedding_matrix
         self.img_dims = self.opt.img_dims
         self.num_classes = self.opt.num_classes
         self.num_hidden_layers = self.opt.num_hidden_layers
    
    def  __init__(self, opt):
          #super(lstm_normI, self).__init__(opt)
          super().__init__(opt)
        
    def text_model(self,dropout_rate):
        question_input = Input(shape=(self.max_seq_length,), dtype='int32' , name='txt_input')
        text = Embedding(input_dim=self.vocab_size, 
                                      output_dim=self.embedding_dim, 
                                      weights=[self.embedding_matrix],
                                      input_length=self.max_seq_length)(question_input)
        
        text = LSTM(512, return_sequences=True)(text)
        text = Dropout(dropout_rate)(text)
        text = LSTM(512)(text)
        text = Dropout(dropout_rate)(text)
        text = Dense(1024, activation='tanh' , name='txt_output')(text)
        
        
        return Model(inputs=question_input, outputs=text)
        
    
        
    def img_model(self,dropout_rate):
         question_img = Input(shape=(224, 224, 3), name='img_input')
         
         
         img = ZeroPadding2D((1,1))(question_img)
         img = Conv2D(64, (3, 3), activation='relu', padding='same')(img)
         img = ZeroPadding2D((1,1))(img)
         img = Conv2D(64, (3, 3), activation='relu')(img)
         img = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(img)
         
         
         img = ZeroPadding2D((1,1))(img)
         img = Conv2D(128, (3, 3), activation='relu', padding='same')(img)
         img = ZeroPadding2D((1,1))(img)
         img = Conv2D(128, (3, 3), activation='relu')(img)
         img = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(img)
         
         
         img = ZeroPadding2D((1,1))(img)
         img = Conv2D(256, (3, 3), activation='relu', padding='same')(img)
         img = ZeroPadding2D((1,1))(img)
         img = Conv2D(256, (3, 3), activation='relu')(img)
         img = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(img)
         
         
         img = ZeroPadding2D((1,1))(img)
         img = Conv2D(512, (3, 3), activation='relu', padding='same')(img)
         img = ZeroPadding2D((1,1))(img)
         img = Conv2D(512, (3, 3), activation='relu')(img)
         img = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(img)
         
         img = ZeroPadding2D((1,1))(img)
         img = Conv2D(512, (3, 3), activation='relu', padding='same')(img)
         img = ZeroPadding2D((1,1))(img)
         img = Conv2D(512, (3, 3), activation='relu')(img)
         img = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(img)
         
         
         img = Flatten()(img)
         
         img = Dense(4096, activation='relu')(img)
         img = Dropout(dropout_rate)(img)
         img = Dense(4096, activation='relu')(img)
         img = Dropout(dropout_rate)(img)
         img = Dense(1024, activation='tanh' , name='img_output')(img)
         
         
         #img = Lambda(lambda  x: K.l2_normalize(x,axis=1), name='img_output')(img)
         
         return Model(inputs=question_img, outputs=img)
         
         
    def build(self):
        
         print('Build LSTM Norm I model...')
         dropout_rate = 0.2
         
         #VQA model
         lstm_model = self.text_model(dropout_rate)
         vgg_model = self.img_model(dropout_rate)
         
         
         #vgg_model.summary()
         #lstm_model.summary()
         
       
         
         mergedOut = Multiply()([lstm_model.get_layer('txt_output').output,vgg_model.get_layer('img_output').output])
         #mergedOut = vgg_model.get_layer('img_output').output
         mergedOut = Dropout(dropout_rate)(mergedOut)
         
         for i in range(self.num_hidden_layers):
             mergedOut = Dense(1024, activation='tanh')(mergedOut)
             mergedOut = Dropout(dropout_rate)(mergedOut)
         #mergedOut = Dense(self.num_classes, activation='sigmoid')(mergedOut) # This is for multi-label classification
         mergedOut = Dense(self.num_classes, activation='softmax')(mergedOut)
         
         newModel = Model([lstm_model.get_layer('txt_input').output,vgg_model.get_layer('img_input').output], mergedOut)
         #newModel = Model(vgg_model.get_layer('img_input').output, mergedOut)
         
         return newModel
  
    

   