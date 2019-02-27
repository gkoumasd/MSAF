import pandas as pd
from functions.load_files import *
from classes import text_based_classes as t
from keras.preprocessing.sequence import pad_sequences
from classes.DottableDict import DottableDict
from functions.mutlilabel_metrics import *
from classes.DataGeneratorPredict import DataGeneratorPredict 

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



model_params =  DottableDict({'name': 'lstm_normI_DAQUAR.h5',
                'custrom_objects': {'precision':precision, 'recall': recall, 'fmeasure': fmeasure},
               })


model = loadmodel(model_params.name,  model_params.custrom_objects) 
model.summary()



ids = test['IDs']
labels = test.iloc[:,1:69]
images = test['imgs']

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
                 'shuffle': True}

generator = DataGeneratorPredict(ids, labels, images, questions,  **data_params)


steps = len(ids) / 32
predictions = model.predict_generator(generator, steps)

print(predictions[0])
print(labels.loc[0])

print(predictions[1])
print(labels.loc[1])

print(predictions[2])
print(labels.loc[2])


predicted_classes = convert_to_class(predictions)



