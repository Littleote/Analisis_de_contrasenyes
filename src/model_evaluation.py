# -*- coding: utf-8 -*-
"""
@author: david
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

class ModelEvaluation:
    def evaluate(pipe, dades, objectiu, name, **evaluacio):
        x = dades
        y = objectiu
        w = np.zeros(len(y))
        pred = np.zeros(len(y))
        classes = np.sort(np.unique(y))
        for c in classes:
            w[y==c] = 1 / sum(y==c)
        kFolds = evaluacio.get('kFold', 5)
        use_weights = evaluacio.get('class_weighted', True)
        kf = KFold(n_splits=kFolds)
        for ind_train, ind_test in kf.split(y):
            x_t, y_t, w_t = x[ind_train], y[ind_train], w[ind_train]
            x_cv = x[ind_test]
            if use_weights:
                pipe.fit(x_t, y_t, model__sample_weight=w_t)
            else:
                pipe.fit(x_t, y_t)
            pred[ind_test] = pipe.predict(x_cv)
        pred = pipe.predict(dades)
        plots = evaluacio.get('plot', [])
        if not type(plots) == list:
            plots = [plots]
        for plot in plots:
            if plot == 'confusion':
                cm = confusion_matrix(y, pred)
                plt.subplots(figsize=(10, 6))
                sns.heatmap(cm, annot = True, fmt = 'g')
                plt.xlabel("Predit")
                plt.ylabel("Real")
                plt.title(f"Matriu de Confusió pel model {name}")
                plt.show()
            elif plot == 'percentage':
                cm = confusion_matrix(y, pred, sample_weight=w)
                plt.subplots(figsize=(10, 6))
                sns.heatmap(cm, annot = True, fmt = 'g')
                plt.xlabel("Predit")
                plt.ylabel("Real")
                plt.title(f"Matriu dels percentatges pel model {name}")
                plt.show()
            elif plot == 'AUC':
                plt.figure(figsize=(15,10))
                ax = plt.gca()
                for c in classes:
                    yi = np.copy(y)
                    yi[yi!=c] = -1
                    yi[yi==c] = 1
                    predi = np.copy(pred)
                    predi[predi!=c] = -1
                    predi[predi==c] = 1
                    PrecisionRecallDisplay.from_predictions(yi, predi, sample_weight=w,\
                            ax=ax, name=f'Precision-recall curve of class {c}')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.legend(loc="lower left")
                plt.title('Precision-Recall Curve')
                plt.show()
            elif plot == 'ROC':
                plt.figure(figsize=(15,10))
                ax = plt.gca()
                for c in classes:
                    yi = np.copy(y)
                    yi[yi!=c] = -1
                    yi[yi==c] = 1
                    predi = np.copy(pred)
                    predi[predi!=c] = -1
                    predi[predi==c] = 1
                    RocCurveDisplay.from_predictions(yi, predi, sample_weight=w,\
                            ax=ax, name=f'ROC curve of class {c}')
                plt.xlabel('False Positive')
                plt.ylabel('True Positive')
                plt.legend(loc="lower right")
                plt.title('ROC Curve')
                plt.show()
            else:
                print(f'Plot for {plot} not implemented.')
                
        scores = evaluacio.get('score', [])
        if not type(plots) == list:
            scores = [scores]
        for score in scores:
            if score == 'all':
                print(classification_report(y, pred))
            elif score == 'accuracy':
                print(f'Accuracy = {sum(y==pred) / len(y)} : {sum(y==pred)}/{len(y)}')
                print(f'Macro accuracy = {sum([sum(c==pred[y==c]) / sum(y==c) for c in classes]) / len(classes)}')
            elif score == 'class accuracy':
                for c in classes:
                    ind = y==c
                    print(f'Accuracy of class {c} = {sum(c==pred[ind]) / sum(ind)} : {sum(c==pred[ind])}/{sum(ind)}')
            else:
                print(f'Score for {score} not implemented.')