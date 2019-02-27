from classes.DottableDict import DottableDict
import models
import pandas as pd
from classes import text_based_classes as t
import pickle
from keras.utils import plot_model
from classes.DataGenerator2 import DataGenerator 
from keras.preprocessing.sequence import pad_sequences
from functions.mutlilabel_metrics import *
from functions.optimizers import *
from functions.load_files import *
from keras.preprocessing.text import Tokenizer, one_hot
import numpy as np



#Load data
train = pd.read_csv('data/train.csv', sep=',')
train = train.rename(columns={ train.columns[0]: "IDs" })
questions = train['questions']

valid = pd.read_csv('data/valid.csv', sep=',')
valid = valid.rename(columns={ valid.columns[0]: "IDs" })
v_questions = valid['questions']

     


tokenizer = load_pickle('data/tokenizer.pickle')
word_index = load_pickle('data/word_index.pickle')
  
vocab_size = len(word_index)

#Calculate max of sequaence words    
#df_train = pd.read_csv('data/train.csv', sep=',')
max_seq_length = 0   
for item in questions:    
    sentence_words = len(item.split())
    if sentence_words> max_seq_length:
        max_seq_length = sentence_words
        
#Embedding
embedding_matrix = t.embedding_matrix(word_index)         
embedding_dim = 300        





params = DottableDict({"network_type": "lstm_normI",
                       "max_seq_length": max_seq_length,
                       "vocab_size": vocab_size,
                       "embedding_dim": embedding_dim,
                       "embedding_matrix": embedding_matrix,
                       "img_dims": (224,224,1),
                       "num_classes": 462})
    
    

model = models.setup(params)    
my_model = model.getModel()
my_model.summary()


#loss_func = 'binary_crossentropy' #This is for multi-label classification
loss_func = 'categorical_crossentropy' #multi-class classification
optimizer = Optimizer('adam')
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

metric_func =['accuracy']
#metric_func = [precision, recall, fmeasure]

my_model.compile(loss = loss_func, optimizer = optimizer, metrics=metric_func)




ids = train['IDs']
ids = np.array(ids)
labels = train.iloc[:,1:463]
labels = np.array(labels)
images = train['imgs']
images = np.array(images)

questions = tokenizer.texts_to_sequences(questions)
questions = pad_sequences(questions, maxlen= max_seq_length) 



# Parameters
data_params = {'batch_size': 32,
               'dim': (224,224),
               'txt_dim': max_seq_length,
               'dir_imgs': 'data/DAQUAR/nyu_depth_images/',
               'n_channels': 3,
               'n_classes': 462,
               'shuffle': True}


#data_params = {'dim': (224,224),
#               'batch_size': 32,
#               'n_classes': 3120,
#               'n_channels': 1,
#               'shuffle': True,
#               'dir_imgs': 'data/vqa2018/VQAMed2018Train/VQAMed2018Train-images/'}
#              

train_generator = DataGenerator(ids, labels, images, questions,  **data_params)



v_ids = valid['IDs']
v_ids = np.array(v_ids)
v_labels = valid.iloc[:,1:463]
v_labels = np.array(v_labels)
v_images = valid['imgs']
v_images = np.array(v_images)

v_questions = tokenizer.texts_to_sequences(v_questions)
v_questions = pad_sequences(v_questions, maxlen= max_seq_length) 

v_data_params = {'batch_size': 32,
                 'dim': (224,224),
                 'txt_dim': max_seq_length,
                 'dir_imgs': 'data/DAQUAR/nyu_depth_images/',
                 'n_channels': 3,
                 'n_classes': 462,
                 'shuffle': True}


# Parameters
#v_data_params = {'dim': (224,224),
#                 'batch_size': 32,
#                 'n_classes': 3120,
#                 'n_channels': 1,
#                 'shuffle': True,
#                 'dir_imgs': 'data/vqa2018/VQAMed2018Valid/VQAMed2018Valid-images/'}
              

valid_generator = DataGenerator(v_ids, v_labels, v_images, v_questions,  **v_data_params)




model_params =  DottableDict({"epochs": 100,
                              "verbose": 1,
                              "workers": 6,
                              "use_multiprocessing": True,
                              "save_h5": 'cnn_simple.h5'})


history = my_model.fit_generator(generator=train_generator,
                    validation_data=valid_generator,
                    use_multiprocessing=model_params.use_multiprocessing,
                    epochs=model_params.epochs,
                    verbose = model_params.verbose,
                    workers=model_params.workers)


#hisrory(history)

my_model.save(model_params.save_h5)


