# -*- coding: utf-8 -*-
"""
@author: david
"""

import numpy as np
from src.clean_data import DataCleaner as dc

class GenerateFeatures:
    IniciDeContrasenya = 'inici'
    CaracterSenseTipus = 'Cap'
    
    def __init__(self, funcs, ferRecompte=True):
        self.funcs = funcs
        self.ferRecompte = ferRecompte
    
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
    
    def mitja_caracters_prob(contrasenya):
        gf = GenerateFeatures
        res = 0
        for c in contrasenya:
            res += np.log(gf.ind.get(c, 1))
        return np.exp(res / len(contrasenya)) / gf.ind_total if not contrasenya == "" else 1
    

    def sequencia_prob(contrasenya):
        gf = GenerateFeatures
        res = 1
        c_ = gf.IniciDeContrasenya
        for c in contrasenya:
            if c not in dc.CaractersValids:
                continue
            res *= gf.pred[c_][c] / gf.pred_total[c_]
            c_ = c
        return res
    
    def mitja_sequencia_prob(contrasenya):
        gf = GenerateFeatures
        res = 1
        c_ = gf.IniciDeContrasenya
        for c in contrasenya:
            if c not in dc.CaractersValids:
                continue
            res *= gf.pred[c_][c] / gf.pred_total[c_]
            c_ = c
        return res ** (1 / len(contrasenya)) if not contrasenya == "" else 1

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
    
    def mitja_cadena(contrasenya):
        gf = GenerateFeatures
        canvis = 0
        ultim = gf.CaracterSenseTipus
        for c in contrasenya:
            actual = dc.TipusCaracters.get(c, gf.CaracterSenseTipus)
            if not actual == ultim:
                ultim = actual
                canvis += 1
        return len(contrasenya) / canvis if not canvis == 0 else 0
    
    def ponderacio_cadena(contrasenya):
        gf = GenerateFeatures
        ultim = gf.CaracterSenseTipus
        S, s = 0, 1
        for c in contrasenya:
            actual = dc.TipusCaracters.get(c, gf.CaracterSenseTipus)
            if not actual == ultim:
                ultim = actual
                s = 1
            else:
                s += 2
            S += s
        return S / len(contrasenya) if not contrasenya == "" else 0
    
    def teMinuscules(contrasenya):
        return any(c in dc.Minuscules for c in contrasenya)
    
    def teMajuscules(contrasenya):
        return any(c in dc.Majuscules for c in contrasenya)
    
    def teXifres(contrasenya):
        return any(c in dc.Xifres for c in contrasenya)
    
    def teEspecials(contrasenya):
        return any(c in dc.Especials for c in contrasenya)
    
    def flagCaracters(contrasenya):
        gf = GenerateFeatures
        return gf.teMinuscules(contrasenya) + 2 * gf.teMajuscules(contrasenya) + \
            4 * gf.teXifres(contrasenya) + 8 * gf.teEspecials(contrasenya)
    
    def get_attrib(contrasenyes, func):
        return np.vectorize(func)(contrasenyes)
    
    def get_attribs(contrasenyes, funcs):
        return np.array([np.vectorize(func)(contrasenyes) for func in funcs])
    
    def fit(self, X, y=None, **args):
        if self.ferRecompte:
            gf = GenerateFeatures
            gf.recompte(X)
        
    def transform(self, X, **args):
        gf = GenerateFeatures
        return gf.get_attribs(X, self.funcs).transpose()
    
    def fit_transform(self, X, y=None, **args):
        self.fit(X)
        return self.transform(X)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import seaborn as sns
    
    gf = GenerateFeatures
    
    dataset = dc.loadCleanData(data='../data/data.csv', regenerate=False)
    contrasenyes = dataset.values[:,0]
    gf.recompte(contrasenyes)
    
    plt.subplots(figsize=(20, 18))
    plt.bar(gf.ind.keys(), gf.ind.values())
    plt.title("Frequencies dels caracters")
    plt.show()
    
    plt.subplots(figsize=(20, 18))
    sns.heatmap([[e for e in d.values()] for d in gf.pred.values()], \
                xticklabels=dc.CaractersValids, yticklabels=gf.pred.keys(), \
                square=True, norm=LogNorm())
    plt.xlabel("Segon")
    plt.ylabel("Primer")
    plt.title("Frequencies de les sequencies")
    plt.show()
    
    plt.xscale('log')
    plt.yscale('log')
    plt.hist(gf.get_attrib(contrasenyes, gf.caracters_prob), \
             bins = np.e ** -np.linspace(200, 00, 70, endpoint=True))
    plt.title("Probabilitat dels caracters")
    plt.show()
    plt.xscale('log')
    plt.yscale('log')
    plt.hist(gf.get_attrib(contrasenyes, gf.mitja_caracters_prob), \
             bins = np.e ** -np.linspace(12, 00, 70, endpoint=True))
    plt.title("Probabilitat mitja dels caracters")
    plt.show()
    plt.xscale('log')
    plt.yscale('log')
    plt.hist(gf.get_attrib(contrasenyes, gf.sequencia_prob), \
             bins = np.e ** -np.linspace(200, 00, 70, endpoint=True))
    plt.title("Probabilitat de les sequencies")
    plt.show()
    plt.xscale('log')
    plt.yscale('log')
    plt.hist(gf.get_attrib(contrasenyes, gf.mitja_sequencia_prob), \
             bins = np.e ** -np.linspace(10, 00, 70, endpoint=True))
    plt.title("Probabilitat mitja de les sequencies")
    plt.show()
    plt.xscale('log')
    plt.yscale('log')
    plt.hist(gf.get_attrib(contrasenyes, gf.aleatorietat), \
             bins = np.e ** np.linspace(-80, 60, 70, endpoint=True))
    plt.title("Que tan barrejats es troben els caracters")
    plt.show()
    plt.yscale('log')
    plt.hist(gf.get_attrib(contrasenyes, gf.llargada), \
             bins = np.linspace(1, 50, 50, endpoint=True))
    plt.title("Longitud de la contrasenya")
    plt.show()
    
    plt.yscale('log')
    plt.hist(gf.get_attrib(contrasenyes, gf.mitja_cadena), \
             bins = np.linspace(1, 10, 70, endpoint=True))
    plt.title("Mitja de la longitud de cada seccio de caracters")
    plt.show()
    plt.yscale('log')
    plt.hist(gf.get_attrib(contrasenyes, gf.ponderacio_cadena), \
             bins = np.linspace(1, 30, 70, endpoint=True))
    plt.title("Mitja ponderada de la longitud de cada seccio de caracters")
    plt.show()
    plt.yscale('log')
    plt.hist(gf.get_attrib(contrasenyes, gf.flagCaracters), \
             bins = np.linspace(0, 15, 16, endpoint=True))
    plt.xlabel("1-Minuscula, 2-Majuscula, 4-Xifra, 8-Especial")
    plt.title("Flags dels tipus de caracters que contenen")
    plt.show()