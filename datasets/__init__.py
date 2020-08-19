# -*- coding: utf-8 -*-

import os
from dataset.multimodal.data_reader import CMUMOSEIDataReader, CMUMOSIDataReader, POMDataReader, IEMOCAPDataReader

def setup(opt):
    dir_path = os.path.join(opt.datasets_dir,opt.data_dir)
    if opt.dataset_name.lower() == 'cmumosei':
        reader = CMUMOSEIDataReader(dir_path,opt)
    elif opt.dataset_name.lower() == 'cmumosi':
        reader = CMUMOSIDataReader(dir_path,opt)
    elif opt.dataset_name.lower() == 'pom':    
        reader = POMDataReader(dir_path,opt)
    elif opt.dataset_name.lower() == 'iemocap':    
        reader = IEMOCAPDataReader(dir_path,opt)
    return reader