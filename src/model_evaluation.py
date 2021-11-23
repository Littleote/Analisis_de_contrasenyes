# -*- coding: utf-8 -*-
"""
@author: david
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

class ModelEvaluation:
    def evaluate(model, dades, objectiu):
        predit = model.predict(dades)
        cm = confusion_matrix(objectiu, predit)
        plt.subplots(figsize=(10, 6))
        sns.heatmap(cm, annot = True, fmt = 'g')
        plt.xlabel("Predit")
        plt.ylabel("Real")
        plt.title("Matriu de Confusió")
        plt.show()
        
        print(f"Accuracy: {accuracy_score(objectiu, predit)}")