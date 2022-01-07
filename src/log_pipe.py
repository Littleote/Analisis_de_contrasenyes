# -*- coding: utf-8 -*-
"""
@author: david
"""

import numpy as np

class LogPipe:
    err = np.float64(1e-200)
    def __init__ (self, columns):
        self.column_transform = columns
    
    def fit (self, X, **args):
        return self
    
    def transform(self, X, copy=True):
        if copy == True:
            X_tr = np.copy(X)
        else:
            X_tr = X
        X_tr[:, self.column_transform] = np.log(X_tr[:, self.column_transform].astype(np.float64) + LogPipe.err)
        return X_tr
    
    def fit_transform(self, X, y=None, copy=True, **args):
        return self.transform(X, copy)