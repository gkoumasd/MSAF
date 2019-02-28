import pickle
import cv2
from keras.preprocessing.image import  img_to_array
from keras.models import load_model

def load_pickle(file):
    with open(file, 'rb') as handle:
        pickle_file = pickle.load(handle)
        return pickle_file
    
    
def save_to_pickle(filename, data):
     with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)  
        
def loadmodel(_model, _custrom_objects= None):

    #Load training model
    model = load_model(_model, _custrom_objects)
    #model.summary()  

    return model   

    


def load_images(dim, img_path):
    
    image = cv2.imread(img_path)
		
    if image is not None:
        image = cv2.resize(image, dim) #224x224
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #print(image.shape)
        image = img_to_array(image)
        #Scale the raw pixel to the range [0,1]
        image = image / 255.0
    else:
        print(img_path , " not loaded")
            
    return image       
    

