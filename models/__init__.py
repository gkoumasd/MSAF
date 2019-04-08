from models.baseline.image import uni_image
from models.baseline.text import uni_text
from models.baseline.multimodal import vgg_cnn

def setup(opt):
    
    print("network type: " + opt.network_type)
    
    if opt.network_type == "multimodal":
        model = vgg_cnn(opt) 
    
    if opt.network_type == "text_based":
        model = uni_text(opt)   
    
    if opt.network_type == "image_based":
        model = uni_image(opt) 
        
     
    
    return  model
