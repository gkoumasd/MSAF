from models.baseline.lstm_normI import lstm_normI
from models.baseline.baseline import vgg_cnn

def setup(opt):
    
    print("network type: " + opt.network_type)
    
    if opt.network_type == "baseline":
        model = vgg_cnn(opt)  
    return  model
    
    if opt.network_type == "lstm_normI":
        model = lstm_normI(opt)  
    return  model
    



