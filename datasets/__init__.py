# -*- coding: utf-8 -*-

def setup(opt):
    if opt.dataset_name.lower() == 'cmumosei':
        from dataset.mosei_reader import CMUMOSEIReader as MMDataReader
    elif opt.dataset_name.lower() == 'cmumosi':
        from dataset.mosi_reader import CMUMOSIReader as MMDataReader
    elif opt.dataset_name.lower() == 'iemocap':
        from dataset.iemocap_reader import IEMOCAPReader as MMDataReader
    else:
        #Default
        from dataset.mosi_reader import CMUMOSIReader as MMDataReader

    reader = MMDataReader(opt)
    return reader