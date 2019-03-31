#This class take an argument like 

#{"network_type": "SimpleVisualCNN",
#          "dims": (250,250,3),
#          "num_classes": 81}

#and converts to a dot type variables, e.g., opt.network_type

class DottableDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
        self.allowDotting()
    def allowDotting(self, state=True):
        if state:
            self.__dict__ = self
        else:
            self.__dict__ = dict()

# -*- coding: utf-8 -*-

