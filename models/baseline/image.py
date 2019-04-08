#VGG19 + multichanel CNN
from keras.applications import VGG16, VGG19, InceptionV3, Xception
from keras.models import Model
from keras.layers import Lambda,Multiply,Concatenate, Input,Dense, Flatten, Dropout, Conv1D,GlobalAveragePooling2D, Conv2D,LSTM,Embedding, MaxPooling1D, MaxPooling2D, ZeroPadding2D, concatenate
from models.BasicModel import BasicModel
import keras.backend as K

import pandas as pd
import pickle


class uni_image(BasicModel):
    def initialize(self):
        
        self.img_dims = self.opt.img_dims
        
        
        
    def  __init__(self, opt):
        #super(lstm_normI, self).__init__(opt)
        super().__init__(opt)
        
   
    
    
        
    def img_model(self):

        img_input = Input(shape=self.img_dims, name='img_input')
        
        #Block 1
        img = Conv2D(32, (3, 3) ,padding='same' , activation='relu')(img_input)
        img = Conv2D(32, (3, 3) ,padding='same', activation='relu')(img)
        img = MaxPooling2D(pool_size=(2, 2))(img)
        img = Dropout(0.25)(img)
        
        #Block 2
        img = Conv2D(64, (3, 3) ,padding='same', activation='relu')(img)
        img = Conv2D(64, (3, 3) ,padding='same', activation='relu')(img)
        img = MaxPooling2D(pool_size=(2, 2))(img)
        img = Dropout(0.25)(img)
        
        #Block 3
        img = Conv2D(64, (3, 3) ,padding='same', activation='relu')(img)
        img = Conv2D(64, (3, 3) ,padding='same', activation='relu')(img)
        img = MaxPooling2D(pool_size=(2, 2))(img)
        img = Dropout(0.25)(img)
        
    
        #Block flatten
        img = Flatten()(img)
        img = Dense(512, activation='relu')(img)
        img = Dropout(0.5, name='img_output')(img)
       
        return Model(inputs=img_input, outputs=img)
    
    
    
    def img_tunning_model(self):
        print('VVG16')
        vgg19_model  = VGG19(weights='imagenet', include_top=False, input_shape = self.img_dims)
        x = vgg19_model.output
        #x = GlobalAveragePooling2D()(x)
        # add fully-connected layer
        #x = Dense(512, activation='relu')(x)
        #x = Dropout(0.3)(x)
        
        #VGG19 top
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dense(2048, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        
        
        x = Dropout(0.5, name='img_output')(x)
         
         
        return Model(inputs=vgg19_model.input, outputs=x)
       
   
        
        
    def build(self):    
        
         print('Build VGG19...')
         
         
         img_model = self.img_model()
         mergedOut = img_model.get_layer('img_output').output
        
        
         mergedOut = Dense(2, activation='softmax' )(mergedOut) 
         
         
         model = Model(img_model.get_layer('img_input').output, mergedOut)
         
         
          #This is for fine tunning
         
         # check the layers by name
         #for i,layer in enumerate(multimodal.layers):
         #    print(i,layer.name)
         
         #layer_num = len(multimodal.layers)
         #print('Total layers:', layer_num)
         #for layer in multimodal.layers[:21]:
         #    layer.trainable = False
         #    print(layer.name)
         
         
         return model