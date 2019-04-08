#This class is all about changing the line loading the entire dataset at once.

import numpy as np
import keras
import cv2
import os
from keras.preprocessing.image import load_img, img_to_array


class DataGeneratorEvaluate(keras.utils.Sequence):
    
    
    
    def __init__(self, list_IDs, labels, images, lw, pw, nw, tetxs, batch_size=32, txt_dim =10, dim=(32,32,32), n_channels=1,
                 dir_imgs = None ,  shuffle=True , nn_type = ''):
        #'Initialization'
        #dim express volume of data. E.g., images of size 224x224 has dim=(224,224) and n_channels=3
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.txt_dim = txt_dim
        self.images = images
        self.lw = lw
        self.pw = pw
        self.nw = nw
        self.tetxs = tetxs
        self.dir_imgs = dir_imgs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.nn_type = nn_type
        self.on_epoch_end()
    
    #Updates indexes after each epoch'
    #the method on_epoch_end is triggered once at the very beginning as well as at the end of
    #each epoch. 
    def on_epoch_end(self):
      self.indexes = np.arange(len(self.list_IDs))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)
          
          
    def load_img_v2(self, img_path):
    
       
        image = cv2.imread(img_path)
		
        if image is not None:
            image = cv2.resize(image, self.dim) #224x224
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #print(image.shape)
            image = img_to_array(image)
        #Scale the raw pixel to the range [0,1]
            image = image / 255.0
        else:
            print(img_path , " not loaded->")
            
        return image      
          
          
    #def load_img(self, img_path):
        #load images, convert them to numpy array, and normalize them
        #print('0k1')
        #img = load_img(img_path, target_size = self.dim)
        #print('Ok2')
       # x = img_to_array(img)
        #Normalize values from 0 to 1
       # x /= 255
        
        #return x      
    
    
    #Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # list_IDs_temp ->the list of IDs of the target batch
    def __data_generation(self, list_IDs_temp):
        # Initialization
        
        #Texts
        X1 = np.empty((self.batch_size, self.txt_dim))
        #Images
        X2 = np.empty((self.batch_size, *self.dim, self.n_channels))
        #Context
        #X3 = np.empty((self.batch_size, 1))
        #X4 = np.empty((self.batch_size, 1))
        
        # This is for binary classification
        y = np.empty((self.batch_size), dtype=int)
        #y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            
            # integer encode the documents
            X1[i,] = self.tetxs[ID]
            
            #Context
            #X3[i,] = self.lw[ID]
            #X4[i,] = self.nw[ID]
            
            y[i,] = self.labels[ID]
            
            
                
            if self.dir_imgs != None:
                
               
                img = self.images[ID].replace('\\','/')
                if (y[i,]==1):
                    subfolder = 'positive/'
                else:
                    subfolder = 'negative/'
                    
                img =  img + '.jpg'
                
                img_path = os.path.join(self.dir_imgs, subfolder, img)
                
                
                
                #print(img_path)
                #X[i,] = np.load('data/' + ID + '.npy')
                
                
                #load_img(img_path)
                
                X2[i,] = self.load_img_v2(img_path)
                
                
                
                
            # Store class into a list
            
            #This is for binaray classification
            
            
        if (self.nn_type=='text_based'):
            X = [X1]
        elif(self.nn_type=='img_based'):
            X = [X2]
        else:
            X = [X1,X2]
        #X = [X1,X1,X1,X2]
        #X = X1
        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, keras.utils.to_categorical(y, num_classes=2)
    
    #Denotes the number of batches per epoch := total samples / batch size 
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    #Generate one batch of data
    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        
       

        return X, y#

# -*- coding: utf-8 -*-

