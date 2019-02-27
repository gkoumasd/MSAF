#Simple CNN
from keras.models import Model,Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, GlobalAveragePooling2D
from models.BasicModel import BasicModel
import keras.backend as K

import pandas as pd#
import pickle

class baselineCNN(BasicModel):
    def initialize(self):
         print('Initialize...')
         self.max_seq_length = self.opt.max_seq_length
         self.vocab_size = self.opt.vocab_size
         self.embedding_dim = self.opt.embedding_dim
         self.embedding_matrix = self.opt.embedding_matrix
         self.img_dims = self.opt.img_dims
         self.num_classes = self.opt.num_classes
    
    def  __init__(self, opt):
          #super(lstm_normI, self).__init__(opt)
          super().__init__(opt)
        
   
         
         
    def build(self):
        
        print('A simple CNN model')

        model = Sequential()  
    

        model.add(Conv2D(32, (3, 3), activation='relu', padding='same',input_shape=(224 , 224, 3)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25)) 	

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        #model.add(Activation('sigmoid')) #multi-label
        model.add(Activation('softmax'))

        # let's train the model using SGD + momentum (how original).
        #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
        #model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=[fmeasure,recall,precision])
        #model.summary() 
        
        return model
             
    
  
    

   