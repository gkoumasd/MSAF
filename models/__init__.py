from models.baseline.lstm_normI import lstm_normI

def setup(opt):
    
    print("network type: " + opt.network_type)
    
    if opt.network_type == "lstm_normI":
        model = lstm_normI(opt)  
    return  model
    



