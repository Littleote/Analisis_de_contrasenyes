# -*- coding: utf-8 -*-
"""
@author: david
"""

import os.path
import pandas as pd

class DataCleaner:
    Minuscules = [chr(c) for c in range(ord('a'), ord('z') + 1)]
    Majuscules = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
    Xifres = [str(i) for i in range(10)]
    Especials = ['.', ';', '-', '_', '+', '*', '<', '>', '[', ']', '{', '}', \
                 '(', ')', '@', '#', '$', '%', '&', '/', '\\', '?', '!', '=', \
                 '^', '~', ' ']
    CaractersValids = Minuscules + Majuscules + Xifres + Especials
    
    def isValid(contrasenya):
        try:
            # Treure les contrasenyes amb caracters que no consideri valids
            #   ja que no importa el format de descodificació especificat,
            #   Python es incapaç de llegir correctament tots els diferents caràcters
            #   caracters i sempre en surten de l'estil '\x03', '\x0f', '\x8d'
            #   o també §, ¶, ª­, þ, ¤, ...
            return all(c in DataCleaner.CaractersValids for c in contrasenya)
        except:
            # Truere les contrasenyes que Pandas converteixi continuament a float
            #   tot i que s'ha marcat la columna de passwords com strings
            return False
        
    def cleanData(data, output):
        assert os.path.isfile(data)
        # Saltarse les files on les dades no estiguin ben formategades
        dataset = pd.read_csv(data, on_bad_lines='skip', encoding='utf-8')
        # Treure les dades que contenen caracters que no acceptem
        dataset = dataset[dataset.apply(lambda s: DataCleaner.isValid(s['password']), axis=1)]
        # Guardar el nou dataset a un fitxer apart
        dataset.to_csv(output, index=False)

if __name__ == "__main__":
    DataCleaner.cleanData('../data/data.csv', '../data/clean_data.csv')