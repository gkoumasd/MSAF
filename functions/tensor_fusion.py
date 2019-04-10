from numpy import newaxis
from keras import backend as K

def outer_product(inputs):
    
    """
    inputs: list of two tensors (of equal dimensions, 
        for which you need to compute the outer product
    """
    
    x, y = inputs
    batchSize = K.shape(x)[0]
    
    outerProduct = x[:,:, newaxis] * y[:,newaxis,:]
    outerProduct = K.reshape(outerProduct, (batchSize, -1))
    
    # returns a flattened batch-wise set of tensors
    return outerProduct