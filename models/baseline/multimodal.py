#VGG19 + multichanel CNN
from keras.applications import VGG16, VGG19, InceptionV3, Xception
from keras.models import Model
from keras.layers import Lambda,Multiply,Concatenate, Input,Dense, Flatten, Dropout, Conv1D,GlobalAveragePooling2D, Conv2D,LSTM,Embedding, MaxPooling1D, MaxPooling2D, ZeroPadding2D, concatenate
from models.BasicModel import BasicModel
import keras.backend as K
from functions.tensor_fusion import *


import pandas as pd#
import pickle


class vgg_cnn(BasicModel):
    def initialize(self):
        
        #text
        self.max_seq_length = self.opt.max_seq_length
        self.vocab_size = self.opt.vocab_size
        self.embedding_dim = self.opt.embedding_dim
        self.embedding_matrix = self.opt.embedding_matrix
        #image
        self.img_dims = self.opt.img_dims
        
        
        
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
        
        text = Dense(32, activation='tanh' , name='txt_output')(merged)
        
        
        
        
        return Model(inputs=txt_input, outputs=text)
    
    
        
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
        img = Dense(32, activation='relu')(img)
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
       
    def context(self):     
        context_input = Input(shape=(1,)) 
        return context_input
        
        
    def build(self):    
        
         print('Build VGG19 & multi-channel CNN model...')
         
         
         txt_model = self.text_model()
         img_model = self.img_model()
         
         txt = txt_model.get_layer('txt_output').output
         img = img_model.get_layer('img_output').output
         
         
         # L2 normalization
         #txt = Lambda(lambda x: K.l2_normalize(x,axis=1))(txt)
         #img = Lambda(lambda x: K.l2_normalize(x,axis=1))(img)
         
         
         #mergedOut = Concatenate()([txt,img])
         #mergedOut = Dense(100, activation='tanh' , name = 'fc_1')(mergedOut)
         #mergedOut = Dense(32, activation='tanh' , name = 'fc_2')(mergedOut)
         
         #Tensor based fusion
         #tensor 'bilinearProduct' contains the batchwise outer product 
         input_dim = 32
         bilinearProduct = Lambda(outer_product, output_shape=(input_dim**2, ))([txt, img]) 
         
         mergedOut = Dense(100, activation='tanh' , name = 'fc_1')(bilinearProduct)
         mergedOut = Dense(32,  activation='tanh' , name = 'fc_2')(mergedOut)
             
         #Sigmoid + Binary crossentropy    
         #mergedOut = Dense(1, activation='sigmoid')(mergedOut) 
         #softmax  + categorical_crossentropy 
         mergedOut = Dense(2, activation='softmax' )(mergedOut) 
         
         
         
         multimodal = Model([txt_model.get_layer('txt_input').output,img_model.get_layer('img_input').output], mergedOut)
         
         
         
         #This is for fine tunning
         
         # check the layers by name
         #for i,layer in enumerate(multimodal.layers):
         #    print(i,layer.name)
         
         #layer_num = len(multimodal.layers)
         #print('Total layers:', layer_num)
         #for layer in multimodal.layers[:21]:
         #    layer.trainable = False
         #    print(layer.name)
            
         #txt_model.summary()
         #img_model.summary()
         
         return multimodal