# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 17:45:50 2020

@author: jpss8
"""

import numpy as np

class Property: #class for stats analysis
    
    def __init__(self, values):
        self.values = values
        self.std = np.std(self.values)
        self.var = np.var(self.values)
        self.mean = np.mean(self.values)
        self.median = np.median(self.values)
        self.rstd = self.std/self.mean