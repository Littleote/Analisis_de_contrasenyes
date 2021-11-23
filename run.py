# -*- coding: utf-8 -*-
"""
@author: david
"""

from src.clean_data import DataCleaner as dc
from src.generate_features import GenerateFeatures as gf

def main(file, regenerate=True):
    clean_file = 'data/clean_data.csv'
    dc.cleanData(file, clean_file)
    
    gf.recompte('TODO')

if __name__ == "__main__":
    main('data/data.csv', regenerate=False)