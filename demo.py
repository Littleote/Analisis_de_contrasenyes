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

def demo(file, regenerate=True):
    clean_file = 'data/clean_data.csv'
    dataset = dc.loadCleanData(data=file, clean_data=clean_file, regenerate=regenerate)
    contrasenyes = dataset.values[:,0]
    seguretat = dataset.values[:,1].astype(np.int64)
    weights = np.zeros(len(seguretat))
    for c in range(3):
        weights[seguretat==c] = 1 / sum(seguretat==c)
    pipeLongitud = Pipeline([("model", MLongitud())])
    clf_args = {"clf1_max_iter": 1e6, "clf1_solver": "liblinear",\
                "clf1_C": 10, "clf1_penalty": "l2",\
                "clf2_n_estimators": 20, "clf2_max_depth": None,\
                "clf2_criterion": "gini", "clf2_max_features": "log2"}
    pipeCClf = Pipeline([("extraction", gf([gf.mitja_cadena, gf.ponderacio_cadena,\
                                        gf.teMinuscules, gf.teMajuscules, gf.teXifres, gf.teEspecials],\
                          ferRecompte=False)), ("scaling", ss()),\
                         ("model", MCClf(classifier1 = LR, classifier2 = RF, **clf_args))])
    pipeLongitud.fit(contrasenyes, seguretat, model__sample_weight=weights)
    pipeCClf.fit(contrasenyes, seguretat, model__sample_weight=weights)
    
    test = input('Posa una contrasenya per evaluar\n' +\
                 ' * escriure "sortir" per finalitzar\n' +\
                 ' * escriure "valids" per veure llista de caracters valids\n')
    while not test == 'sortir':
        if test == 'valids':
            print("Minuscules: " + " ".join(dc.Minuscules))
            print("Majuscules: " + " ".join(dc.Majuscules))
            print("Xifres: " + " ".join(dc.Xifres))
            print("Especials: " + " ".join(dc.Especials))
        else:
            check = "".join([c for c in test if c in dc.CaractersValids])
            if len(check) != len(test):
                print("La contrasenya conté caracters invalids")
                print(f"S'utilitzarà la contrasenya {check} en comptes de {test}")
            if len(check) < 6:
                print(f"La contrasenya {check} es massa curta")
            else:
                long = pipeLongitud.predict(np.array([check]))[0]
                cclf = pipeCClf.predict(np.array([check]))[0]
                print(f"Model de llargada: {long}")
                print(f"Model sense llargada: {cclf}")
                print(f"Model conjunt (minim): {min(long, cclf)}")
                print(f"Model conjunt (mitja): {(long + cclf) / 2}")
        test = input('Posa una contrasenya per evaluar\n' +\
                     ' * escriure "sortir" per finalitzar\n' +\
                     ' * escriure "valids" per veure llista de caracters valids\n')

if __name__ == "__main__":
    demo('data/data.csv', regenerate=False)