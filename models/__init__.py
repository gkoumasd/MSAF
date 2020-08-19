#
from .multimodal.monologue.MLP import MLP
from .multimodal.monologue.GraphMFN import GraphMFN
from .multimodal.monologue.QDNN import uQDNN
from .multimodal.monologue.QDNNAblation import QDNNAblation
from .multimodal.monologue.LocalMixtureNN import LocalMixtureNN
from .multimodal.dialogue.BCLSTM import BCLSTM
from .multimodal.monologue.CFN import CFN
from .multimodal.monologue.RAVEN import RAVEN
from .multimodal.dialogue.DialogueRNN import BiDialogueRNN
#from .multimodal.dialogue.DialogueGCN import DialogueGCN
from .multimodal.dialogue.ICON import ICON
from .multimodal.dialogue.CMN import CMN
from .classification.DCNN import DCNN
from .multimodal.dialogue.QAttN import QAttN
from .multimodal.dialogue.QMN import QMN
from .multimodal.dialogue.QMNAblation import QMNAblation
from .multimodal.dialogue.CCMF import CCMF
def setup(opt):
    
    print("network type: " + opt.network_type)
    if opt.network_type == "mlp":
        model = MLP(opt) 
    elif opt.network_type == "ef-lstm":
        if opt.dialogue_format:
            from .multimodal.dialogue.EFLSTM import EFLSTM
        else:
            from .multimodal.monologue.EFLSTM import EFLSTM
        model = EFLSTM(opt)
    elif opt.network_type == "tfn":
        if opt.dialogue_format:
            from .multimodal.dialogue.TFN import TFN
        else:
            from .multimodal.monologue.TFN import TFN
        model = TFN(opt)
    elif opt.network_type == "marn":
        if opt.dialogue_format:
            from .multimodal.dialogue.MARN import MARN
        else:
            from .multimodal.monologue.MARN import MARN
        model = MARN(opt)
    elif opt.network_type == "rmfn":
        if opt.dialogue_format:
            from .multimodal.dialogue.RMFN import RMFN
        else:
            from .multimodal.monologue.RMFN import RMFN
        model = RMFN(opt)
    elif opt.network_type == 'lmf':
        if opt.dialogue_format:
            from .multimodal.dialogue.LMF import LMF
        else:
            from .multimodal.monologue.LMF import LMF
        model = LMF(opt)
    elif opt.network_type == 'mfn':
        if opt.dialogue_format:
            from .multimodal.dialogue.MFN import MFN
        else:
            from .multimodal.monologue.MFN import MFN
        model = MFN(opt)
    elif opt.network_type == 'lsthm':
        if opt.dialogue_format:
            from .multimodal.dialogue.LSTHM import LSTHM
        else:
            from .multimodal.monologue.LSTHM import LSTHM
        model = LSTHM(opt)
    elif opt.network_type == 'lf-lstm':
        if opt.dialogue_format:
            from .multimodal.dialogue.LFLSTM import LFLSTM
        else:
            from .multimodal.monologue.LFLSTM import LFLSTM
        model = LFLSTM(opt)
    elif opt.network_type == 'multimodal-transformer':
        if opt.dialogue_format:
            from .multimodal.dialogue.MULT import MULT
        else:
            from .multimodal.monologue.MULT import MULT
        model = MULT(opt)
    elif opt.network_type == 'cfn':
        model = CFN(opt)
    elif opt.network_type == 'raven':
        model = RAVEN(opt)
    else:
        raise Exception("model not supported: {}".format(opt.network_type))
    return model
