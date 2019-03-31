from keras import optimizers

def Optimizer(Optimizer):
    if Optimizer== 'adam':
        optim = optimizers.Adam(lr=0.00025, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    elif Optimizer == 'rmsprop':
        optim = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    elif Optimizer == 'sgd-momentum':
        optim = optimizers.SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True)
    elif Optimizer == 'adagrad':
        optim = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
        
    return optim        
    
    