# -*- coding: utf-8 -*-
"""
@author: david
"""

import numpy as np
from src.clean_data import DataCleaner as dc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler as ss
from src.generate_features import GenerateFeatures as gf

from models.model_longitud import ModelLongitud as MLongitud
from models.model_custom_classifier import ModelCustomClassifier as MCClf
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF

from src.model_evaluation import ModelEvaluation as ME

def main(file, regenerate=True):
    clean_file = 'data/clean_data.csv'
    dataset = dc.loadCleanData(data=file, clean_data=clean_file, regenerate=regenerate)
    contrasenyes = dataset.values[:,0]
    seguretat = dataset.values[:,1].astype(np.int64)
    pipeLongitud = Pipeline([("model", MLongitud())])
    clf_args = {"clf1_max_iter": 1e6, "clf1_solver": "liblinear",\
                "clf1_C": 10, "clf1_penalty": "l2",\
                "clf2_n_estimators": 20, "clf2_max_depth": None,\
                "clf2_criterion": "gini", "clf2_max_features": "log2"}
    pipeCClf = Pipeline([("extraction", gf([gf.mitja_cadena, gf.ponderacio_cadena,\
                                        gf.teMinuscules, gf.teMajuscules, gf.teXifres, gf.teEspecials],\
                          ferRecompte=False)), ("scaling", ss()),\
                         ("model", MCClf(classifier1 = LR, classifier2 = RF, **clf_args))])
    evaluacio = {'kFold': 10, 'class_weighted': True,\
                 'plot': ['confusion', 'percentage', 'AUC', 'ROC'],\
                'score': ['all', 'accuracy', 'class accuracy']}
    ME.evaluate(pipeLongitud, contrasenyes, seguretat, name="Model sobre la longitud", **evaluacio)
    ME.evaluate(pipeCClf, contrasenyes, seguretat, name="Model propi sense la longitud", **evaluacio)

if __name__ == "__main__":
    main('data/data.csv', regenerate=False)