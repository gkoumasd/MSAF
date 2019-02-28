import pandas as pd
from functions.load_files import *
from classes import text_based_classes as t
from keras.preprocessing.sequence import pad_sequences
from classes.DottableDict import DottableDict
from functions.mutlilabel_metrics import *
from classes.DataGeneratorPredict import DataGeneratorPredict
import numpy as np 

test = pd.read_csv('data/test.csv', sep=',')
test = test.rename(columns={ test.columns[0]: "IDs" })
questions = test['questions']

tokenizer = load_pickle('data/tokenizer.pickle')
word_index = load_pickle('data/word_index.pickle')


vocab_size = len(word_index)


# integer encode the documents
max_seq_length = 30   



#Embedding
#embedding_matrix = t.embedding_matrix(word_index)         
#embedding_dim = 300  



#model_params =  DottableDict({'name': 'lstm_normI_DAQUAR.h5',
#                'custrom_objects': {'precision':precision, 'recall': recall, 'fmeasure': fmeasure},
#               })


model_params =  DottableDict({'name': 'lstm_normI_DAQUAR.h5'})
model = load_model(model_params.name)
model.summary()



ids = test['IDs']
ids = np.array(ids)
labels = test.iloc[:,1:463]
labels = np.array(labels)
images = test['imgs']
images = np.array(images)

questions = tokenizer.texts_to_sequences(questions)
questions = pad_sequences(questions, maxlen= max_seq_length) 




#question = questions[:,0:34]
#img_path = 'data/' + images[0] + '.png'
#image = load_images((224,224), img_path)#

#pred = model.predict([question,image])

data_params = {'batch_size': 32,
                 'dim': (224,224),
                 'txt_dim': max_seq_length,
                 'dir_imgs': 'data/DAQUAR/nyu_depth_images/',
                 'n_channels': 3,
                 'n_classes': 462,
                 'shuffle': True}

generator = DataGeneratorPredict(ids, labels, images, questions,  **data_params)



model_params =  DottableDict({"verbose": 1,
                              "workers": 6,
                              "use_multiprocessing": True})

metrics = model.evaluate_generator(generator=generator,
                                    use_multiprocessing=model_params.use_multiprocessing,
                                    verbose = model_params.verbose,
                                    workers=model_params.workers)

print("{}: {}".format(model.metrics_names[0], metrics[0]))
print("{}: {}".format(model.metrics_names[1], metrics[1]))




#steps = len(ids) / 32
#predictions = model.predict_generator(generator, steps)

#print("{}: {}".format(model.metrics_names[0], predictions[0]))


#print(predictions[0])
#print(labels.loc[0])

#print(predictions[1])
#print(labels.loc[1])

#print(predictions[2])
#print(labels.loc[2])



#import numpy as np
#categories = load_pickle('data/category_words.pickle')
#Get lalel
#lalel = labels[187].tolist()
#Find index
#index = [n for n,x in enumerate(lalel) if x==1]
#category = categories[index[0]]




