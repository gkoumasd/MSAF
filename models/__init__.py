#
from .MLP import MLP
from .RAVEN import RAVEN
from .EFLSTM import EFLSTM
from .TFN import TFN
from .MARN import MARN
from .RMFN import RMFN
from .LMF import LMF
from .MFN import MFN
from .LSTHM import LSTHM
from .LFLSTM import LFLSTM
from .MULT import MULT
def setup(opt):
    
    print("network type: " + opt.network_type)
    if opt.network_type == "mlp":
        model = MLP(opt) 
    elif opt.network_type == "ef-lstm":
        model = EFLSTM(opt)
    elif opt.network_type == "tfn":
        model = TFN(opt)
    elif opt.network_type == "marn":
        model = MARN(opt)
    elif opt.network_type == "rmfn":
        model = RMFN(opt)
    elif opt.network_type == 'lmf':
        model = LMF(opt)
    elif opt.network_type == 'mfn':
        model = MFN(opt)
    elif opt.network_type == 'lsthm':
        model = LSTHM(opt)
    elif opt.network_type == 'lf-lstm':
        model = LFLSTM(opt)
    elif opt.network_type == 'multimodal-transformer':
        model = MULT(opt)
    elif opt.network_type == 'raven':
        model = RAVEN(opt)
    else:
        raise Exception("model not supported: {}".format(opt.network_type))
    return model
