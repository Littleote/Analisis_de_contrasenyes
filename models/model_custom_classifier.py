# -*- coding: utf-8 -*-
"""
@author: david
"""

import numpy as np

class ModelCustomClassifier:
    def __init__(self, classifier1, classifier2=None, **classifier_args):
        # Separar arguments que només siguin per un
        clf1_args = {key if not key.startswith('clf1_') else key[5:] : arg\
                     for key, arg in classifier_args.items() if not key.startswith('clf2_')}
        clf2_args = {key if not key.startswith('clf2_') else key[5:] : arg\
                     for key, arg in classifier_args.items() if not key.startswith('clf1_')}
        # Utilitzar el mateix classificador si només s'ha indicat un
        if classifier2 is None:
            classifier2 = classifier1
        self.clf1 = classifier1(**clf1_args)
        self.clf2 = classifier2(**clf2_args)
    
    def fit(self, X, y, sample_weight=None):
        ind = (y<2)
        # Diferenciar entre (0, 1) i (2)
        X1, y1, sample_weight1 = X, np.copy(y), sample_weight
        y1[ind] = 0
        self.clf1.fit(X1, y1, sample_weight=sample_weight1)
        # Diferenciar entre (0) i (1)
        if sample_weight is None:
            X2, y2, sample_weight2 = np.copy(X)[ind], np.copy(y)[ind], None
        else:
            X2, y2, sample_weight2 = np.copy(X)[ind], np.copy(y)[ind], np.copy(sample_weight)[ind]
        self.clf2.fit(X2, y2, sample_weight=sample_weight2)
    
    def predict(self, X, **predict_params):
        pred = self.clf1.predict(X, **predict_params)
        ind = (pred<2)
        if sum(ind) > 0:
            pred[ind] = self.clf2.predict(X[ind], **predict_params)
        return pred