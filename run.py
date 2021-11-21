# -*- coding: utf-8 -*-
"""
@author: david
"""

from src.clean_data import DataCleaner as dc

def main(file, regenerate=True):
    clean_file = 'data/clean_data.csv'
    dc.cleanData(file, clean_file)

if __name__ == "__main__":
    main('data/data.csv', regenerate=False)