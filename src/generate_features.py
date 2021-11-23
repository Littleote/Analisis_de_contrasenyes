# -*- coding: utf-8 -*-
"""
@author: david
"""

import numpy as np
from clean_data import DataCleaner as dc

class GenerateFeatures:
    IniciDeContrasenya = 'inici'
    
    def __recompte(contrasenya):
        gf = GenerateFeatures
        c_ = gf.IniciDeContrasenya
        for c in contrasenya:
            gf.pred[c_][c] += 1
            gf.pred_total[c_] += 1
            gf.ind[c] += 1
            gf.ind_total += 1
            c_ = c
    
    def recompte(contrasenyes):
        gf = GenerateFeatures
        gf.pred = {c:{c:1 for c in dc.CaractersValids} \
                                 for c in dc.CaractersValids +
                                 [gf.IniciDeContrasenya]}
        gf.pred_total = {c:len(dc.CaractersValids)
                                       for c in dc.CaractersValids +
                                       [gf.IniciDeContrasenya]}
        gf.ind = {c:0 for c in dc.CaractersValids}
        gf.ind_total = 0
        for p in contrasenyes:
            gf.__recompte(p)
    
    def caracters_prob(contrasenya):
        gf = GenerateFeatures
        res = 1
        for c in contrasenya:
            res *= gf.ind.get(c, 1)
        return res / (gf.ind_total ** len(contrasenya))
    

    def probabilitat_seq(contrasenya):
        gf = GenerateFeatures
        res = 1
        c_ = gf.IniciDeContrasenya
        for c in contrasenya:
            if c not in dc.CaractersValids:
                continue
            res *= gf.pred[c_][c] / gf.pred_total[c_]
            c_ = c
        return res

    def aleatorietat(contrasenya):
        gf = GenerateFeatures
        res = 1
        c_ = gf.IniciDeContrasenya
        for c in contrasenya:
            if c not in dc.CaractersValids:
                continue
            res *= gf.ind[c] / gf.pred[c_][c] * \
                gf.pred_total[c_] / gf.ind_total
            c_ = c
        return res

    def llargada(contrasenya):
        return len(contrasenya)
    
    def get_attrib(contrasenyes, func):
        return np.vectorize(func)(contrasenyes)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import seaborn as sns
    import pandas as pd
    
    gf = GenerateFeatures
    
    dataset = pd.read_csv('../data/clean_data.csv', on_bad_lines='skip', encoding='utf-8')
    contrasenyes = dataset.values[:,0]
    gf.recompte(contrasenyes)
    
    plt.subplots(figsize=(20, 18))
    plt.bar(gf.ind.keys(), gf.ind.values())
    plt.show()
    
    plt.subplots(figsize=(20, 18))
    sns.heatmap([[e for e in d.values()] for d in gf.pred.values()], \
                xticklabels=dc.CaractersValids, yticklabels=gf.pred.keys(), \
                square=True, norm=LogNorm())
    plt.xlabel("Segon")
    plt.ylabel("Primer")
    plt.title("Sequencies")
    plt.show()
    
    plt.xscale('log')
    plt.yscale('log')
    plt.hist(gf.get_attrib(contrasenyes, gf.caracters_prob), \
             bins = np.e ** -np.linspace(200, 00, 70, endpoint=True))
    plt.show()
    plt.xscale('log')
    plt.yscale('log')
    plt.hist(gf.get_attrib(contrasenyes, gf.probabilitat_seq), \
             bins = np.e ** -np.linspace(200, 00, 70, endpoint=True))
    plt.show()
    plt.xscale('log')
    plt.yscale('log')
    plt.hist(gf.get_attrib(contrasenyes, gf.aleatorietat), \
             bins = np.e ** np.linspace(-80, 60, 70, endpoint=True))
    plt.show()
    plt.yscale('log')
    plt.hist(gf.get_attrib(contrasenyes, gf.llargada), \
             bins = np.linspace(1, 50, 50, endpoint=True))
    plt.show()