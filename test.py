import pandas as pd
from functions.load_files import *
from classes import text_based_classes as t
from keras.preprocessing.sequence import pad_sequences
from classes.DottableDict import DottableDict
from functions.mutlilabel_metrics import *
from classes.DateGeneratorEvaluate import DataGeneratorEvaluate
from classes.DataGeneratorPredict import DataGeneratorPredict
import numpy as np 




test = pd.read_csv('data/test.csv', sep=',')
test = test.replace(np.nan, '', regex=True)
test_txts = test['txt']
test = test.rename(columns={ test.columns[0]: "IDs" })

tokenizer = load_pickle('data/tokenizer.pickle')
word_index = tokenizer.word_index


vocab_size = len(word_index)


# integer encode the documents
max_seq_length = 49   



#Embedding
#embedding_matrix = t.embedding_matrix(word_index)         
#embedding_dim = 300  



#model_params =  DottableDict({'name': 'lstm_normI_DAQUAR.h5',
#                'custrom_objects': {'precision':precision, 'recall': recall, 'fmeasure': fmeasure},
#               })


model_params =  DottableDict({'name': 'trainedModels/img.h5'})
model = load_model(model_params.name)
model.summary()


ids = test['IDs']
ids = np.array(ids)
labels = test['y']
labels = np.array(labels)
images = test['img']
images = np.array(images)
lw = test['txt_lenth']
lw = np.array(lw)
pw = test['pos_words']
pw = np.array(pw)
nw = test['neg_words']
nw = np.array(nw)


test_txts = tokenizer.texts_to_sequences(test_txts)
test_txts = pad_sequences(test_txts, maxlen= max_seq_length) 


#question = questions[:,0:34]
#img_path = 'data/' + images[0] + '.png'
#image = load_images((64,64), img_path)#

#pred = model.predict([question,image])

data_params = {'batch_size': 32,
                 'dim': (64,64),
                 'txt_dim': max_seq_length,
                 'dir_imgs': 'datasets/Flickr/data/flickr-dataset/images/testingset',
                 'n_channels': 3,
                 'shuffle': False}

generator_evalutate = DataGeneratorEvaluate(ids, labels, images, lw,pw,nw, test_txts,  **data_params)



model_params =  DottableDict({"verbose": 1,
                              "workers": 6,
                              "use_multiprocessing": True})

metrics = model.evaluate_generator(generator=generator_evalutate,
                                    use_multiprocessing=model_params.use_multiprocessing,
                                    verbose = model_params.verbose,
                                    workers=model_params.workers)

print("{}: {}".format(model.metrics_names[0], metrics[0]))
print("{}: {}".format(model.metrics_names[1], metrics[1]))


#generator_predict= DataGeneratorPredict(ids, labels, images, questions,  **data_params)

#steps = len(ids) / 32#predictions = model.predict_generator(generator, steps)
#predictions = model.predict_generator(generator_predict, steps)


#prediction = predictions[0]
#print(prediction)
#predictionIndex = np.argmax(prediction, axis=-1)
#print(predictionIndex)
#print(prediction[predictionIndex])

#predictions = np.argmax(predictions, axis=-1)



#categories = load_pickle('data/category_words.pickle')


#with open('data/predictions.txt', 'w') as f:
#    for i in range(len(ids)):
#        prediction = categories[predictions[i]]
#        f.write("%s," % prediction)
#    
#print('Predictions are done!')    


#groundtrouths = np.argmax(labels, axis=-1)
#with open('data/groundtrouth.txt', 'w') as f:
#    for i in range(len(ids)):
#        groundtrouth = categories[groundtrouths[i]]
#        f.write("%s," % groundtrouth)
    
#print('Groundtruths are done!')    




#print(predictions[1])
#print(labels.loc[1])

#print(predictions[2])
#print(labels.loc[2])



#import numpy as np
#
#Get lalel
#lalel = labels[187].tolist()
#Find index
#index = [n for n,x in enumerate(lalel) if x==1]
#category = categories[index[0]]