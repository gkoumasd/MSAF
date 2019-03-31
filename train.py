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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from time import time
from tensorflow.python.keras.callbacks import TensorBoard



#Load data
train = pd.read_csv('data/train.csv', sep=',')
train = train.replace(np.nan, '', regex=True)
train_txts = train['txt']
train = train.rename(columns={ train.columns[0]: "IDs" })


valid = pd.read_csv('data/valid.csv', sep=',')
valid = valid.replace(np.nan, '', regex=True)
valid = valid.rename(columns={ valid.columns[0]: "IDs" })
valid_txts = valid['txt']

     


tokenizer = load_pickle('data/tokenizer.pickle')
word_index = tokenizer.word_index

#This part is only you decide to use words repeated at least 3 times in the corpus
#top_words = load_pickle('data/top_words_texts.pickle')
#tokenizer = Tokenizer(oov_token='UNK')
#tokens = tokenizer.fit_on_texts(top_words)
#vocab_size = len(tokenizer.word_index) + 1
  
vocab_size = len(word_index)

#Calculate max of sequaence words    
#df_train = pd.read_csv('data/train.csv', sep=',')
max_seq_length = 0   
for text in train_txts:    
    sentence_words = len(str(text).split())
    if sentence_words> max_seq_length:
        max_seq_length = sentence_words
        
#Embedding
embedding_matrix = t.embedding_matrix(word_index)         
embedding_dim = 300        





params = DottableDict({"network_type": "baseline",
                       "max_seq_length": max_seq_length,
                       "vocab_size": vocab_size,
                       "embedding_dim": embedding_dim,
                       "embedding_matrix": embedding_matrix,
                       "img_dims": (64,64,3),
                       "num_hidden_layers":2})
    

    
#tensorboard = TensorBoard(
#    log_dir="logs/{}".format(time()),
#    histogram_freq=1,
#    write_images=True)

#callbacks = [
#    tensorboard
#]    

model = models.setup(params)    
my_model = model.getModel()
my_model.summary()


#loss_func = 'binary_crossentropy' #This is for multi-label classification or binary classsification
loss_func = 'categorical_crossentropy' #multi-class classification
optimizer = Optimizer('adam')
sgd = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)

metric_func =['accuracy']
#metric_func = [precision, recall, fmeasure]







my_model.compile(loss = loss_func, optimizer = sgd, metrics=metric_func)

ids = train['IDs']
ids = np.array(ids)
labels = train['y']
#labels = to_categorical( labels, num_classes = 2 )
#labels = labels.astype(int)
labels = np.array(labels)
images = train['img']
images = np.array(images)
lw = train['txt_lenth']
lw = np.array(lw)
pw = train['pos_words']
pw = np.array(pw)
nw = train['neg_words']
nw = np.array(nw)


train_txts = tokenizer.texts_to_sequences(train_txts)
train_txts = pad_sequences(train_txts, maxlen= max_seq_length) 



# Parameters
data_params = {'batch_size': 32,
               'dim': (64,64),
               'txt_dim': max_seq_length,
               'dir_imgs': 'datasets/Flickr/data/flickr-dataset/images/trainingset',
               'n_channels': 3,
               'shuffle': True}


#data_params = {'dim': (64,64),
#               'batch_size': 32,
#               'n_classes': 3120,
#               'n_channels': 1,
#               'shuffle': True,
#               'dir_imgs': 'data/vqa2018/VQAMed2018Train/VQAMed2018Train-images/'}
#              

train_generator = DataGenerator(ids, labels, images, lw,pw,nw, train_txts,  **data_params)



v_ids = valid['IDs']
v_ids = np.array(v_ids)
v_labels = valid['y']
#v_labels = to_categorical( v_labels, num_classes = 2 )
#v_labels = v_labels.astype(int)
v_labels = np.array(v_labels)
v_images = valid['img']
v_images = np.array(v_images)
v_lw = valid['txt_lenth']
v_lw = np.array(v_lw)
v_pw = valid['pos_words']
v_pw = np.array(v_pw)
v_nw = valid['neg_words']
v_nw = np.array(v_nw)

valid_txts = tokenizer.texts_to_sequences(valid_txts)
valid_txts = pad_sequences(valid_txts, maxlen= max_seq_length) 

v_data_params = {'batch_size': 32,
                 'dim': (64,64),
                 'txt_dim': max_seq_length,
                 'dir_imgs': 'datasets/Flickr/data/flickr-dataset/images/trainingset',
                 'n_channels': 3,
                 'shuffle': True}


# Parameters
#v_data_params = {'dim': (64,64),
#                 'batch_size': 32,
#                 'n_classes': 3120,
#                 'n_channels': 1,
#                 'shuffle': True,
#                 'dir_imgs': 'data/vqa2018/VQAMed2018Valid/VQAMed2018Valid-images/'}
              

valid_generator = DataGenerator(v_ids, v_labels, v_images, v_lw,v_pw,v_nw,valid_txts,  **v_data_params)




model_params =  DottableDict({"epochs": 20,
                              "verbose": 1,
                              "workers": 6,
                              "use_multiprocessing": True,
                              "save_h5": 'trainedModels/img.h5'})


history = my_model.fit_generator(generator=train_generator,
                    validation_data=valid_generator,
                    use_multiprocessing=model_params.use_multiprocessing,
                    epochs=model_params.epochs,
                    verbose = model_params.verbose,
                    workers=model_params.workers)


#hisrory(history)




#epochs = range(1, len(history.history['loss']) + 1)
#plt.title("Accuracy")
#plt.plot(epochs,history.history["acc"], color="r", label="train")
#plt.plot(epochs,history.history["val_acc"], color="b", label="validation")
#plt.xlabel('Epochs')
#plt.grid()
#plt.legend(loc='best')
#plt.tight_layout()
#plt.savefig('acc_txt.png')


# plot loss function
plt.subplot(211)
plt.title("Accuracy")
plt.plot(history.history["acc"], color="r", label="train")
plt.plot(history.history["val_acc"], color="b", label="validation")
plt.legend(loc="best")

plt.subplot(212)
plt.title("Loss")
plt.plot(history.history["loss"], color="r", label="train")
plt.plot(history.history["val_loss"], color="b", label="validation")
plt.legend(loc="best")

plt.tight_layout()
plt.savefig('trainedModels/img_accuracy_loss.png')

my_model.save(model_params.save_h5)

