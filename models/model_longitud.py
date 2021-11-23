# -*- coding: utf-8 -*-
"""
@author: david
"""

import numpy as np

class ModelLongitud:
    def predict(X):
        return np.vectorize(lambda x: 0 if len(x) < 8 else 1 if len(x) < 14 else 2)(X)