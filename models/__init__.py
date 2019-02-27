from models.baseline.lstm_normI import lstm_normI
from models.baseline.simplecnn import baselineCNN

def setup(opt):
    
    print("network type: " + opt.network_type)
    
    if opt.network_type == "lstm_normI":
        model = lstm_normI(opt)  
    return  model
    
    if opt.network_type == "simplecnn":
        model = baselineCNN(opt)  
    return  model
    
    
	
	


