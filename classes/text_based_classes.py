import os
import numpy as np

GLOVE_DIR = 'data/glove.6B/'
EMBEDDING_DIM = 300

def embedding_matrix(word_index):
    
    #Preparing the Embedding layer
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'),  encoding="utf8")
    for line in f:
     values = line.split()
     word = values[0]
     coefs = np.asarray(values[1:], dtype='float32')
     embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    #compute our embedding matrix
    embedding_matrix = np.zeros((len(word_index) , EMBEDDING_DIM))
    for word, i in word_index.items():
     embedding_vector = embeddings_index.get(word)
     if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i-1] = embedding_vector

    return embedding_matrix