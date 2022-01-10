# Pràctica Kaggle APC UAB 2021-22
# Analisis_de_contrasenyes
### Nom: David Candela Rubio
### DATASET: Password Strength Classifier Dataset
### URL: [kaggle](https://www.kaggle.com/bhavikbb/password-strength-classifier-dataset)
## Resum
El dataset utilitza contrasenyes del leak de 000webhost i els assigna un nivell de proteccio a partir de tres algoritmes diferents quan aquests coincideixen.
En el dataset tenim dues columnes, la primera conté una contrasenya i la segona el nivell de protecció que ofereix expresat com **0**, **1** o **2**.
### Objectius del dataset
Volem aconseguir una indicació de que tan segura seria una contrasenya.
## Experiments
Aqui es troben dividits en quatre sets els tipus de dades extretes de la contrasenya.
#### set 1
Primer hem buscat atributs per descriure la seguretat a través de probabilitat, com pot ser els caracters o la quantitat que s'han d'adivinar per trobar la contrasenya.
* Probabilitat dels caracters
* Probabilitat de les parelles de caracters
* Que tan barrejats es troben
* Longitud de la cadena

Resultats:  
![Figura-Matriu de percentatges del model longitud](https://user-images.githubusercontent.com/57794658/148817614-03375566-ff00-43ed-8e03-a8f08c0f0969.png)

#### set 1.b
Al veure que la longitud es suficient per trobar els resultats del dataset hem probat a treurel per veure si hi ha alguna altre relació. (Ja que el dataset està fet amb l'intersecció de tres algoritmes)
* Probabilitat mitja dels caracters
* Probabilitat mitja de les parelles de caracters
* Que tan barrejats es troben

Resultats:  
![Figura-Matriu de confusio set 1 b](https://user-images.githubusercontent.com/57794658/148817550-63bbd4d2-56a2-42c1-a602-2b323cb94d70.png)

#### set 2
Longitud mitja de les cadenes dels tipus de caracters. Per exemple: abcdABC12 son una cadena de quatre, una de tres i una de dos.  
* Longitud mitja
* Longitud mitja ponderada

Resultats:  
![Figura-Matriu de confusio set 2](https://user-images.githubusercontent.com/57794658/148817584-0d15b39d-1819-4aea-a395-294e093adc56.png)

#### set 3
Importancia de tenir majuscules, minuscules, xifres i/o caracters especial.
* Contè minuscules
* Contè majuscules
* Contè xifres
* Contè caracters especials

Resultats:  
![Figura-Matriu de confusio set 3](https://user-images.githubusercontent.com/57794658/148817660-172f51a7-a1f9-4ec5-aeca-dd1888fea299.png)

### Preprocessat
Pels valors de la probabilitat i el de la qualitat de la barreja dels caracters hem calculat el seu logaritme i per a tots els atributs hem aplicat [Standard Scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) per normalitzar les dades. 
### Model
Resultats fent Kfolds en cinc parts i entrenat amb només un 10% de les dades:  
| Model | Parametres | Altres | Mètrica | Temps |  
| -- | -- | -- | -- | -- |  
| **Models amb llargada** |  |  |  |  |  
| Doble if | set Llargada | if 1: < 8, if 2: < 14 | 100.00% | 0.0s |  
| **Models sense llargada** |  |  |  |  |  
| [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) | set 1.b, set 2 i set 3 |  | 93.57% | 14.06s |  
| [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) | set 1.b, set 2 i set 3 | PCA: 5 components | 85.90% | 4.71s |  
| [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) | set 2 i set 3 |  | 84.87% | 7.29s |  
| [SVM Lineal](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) | set 1.b, set 2 i set 3 |  | 92.60% | 7min 29.24s |  
| [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) | set 1.b, set 2 i set 3 | Kernel: RBF | 96.42% | 12min 10.99s |  
| [Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) | set 1.b, set 2 i set 3 | Gaussian | 88.90% | 1.98s |  
| [Bagging](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) | set 1.b, set 2 i set 3 | Bagging of: Logistic Regressors, max samples: 0.5, max features: 0.5 | 89.21% | 18.87s |  
| [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) | set 1.b, set 2 i set 3 | trees: 10, max depth: 20 | 98.99% | 3.74s |  
| [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) | set 2 i set 3 | trees: 10, max depth: 20 | 99.45% | 2.28s |
| Simple Neuronal Network | set 1.b, set 2 i set 3 | layers: {9, 32, 10, 3} | 90.15% | 86.4s |  
| Custom Classifier | set 2 i set 3 | Boosting of: Logistic Regressor + Random Forest | 99.55% | 4.21s |  
| **Models sense llargada optimitzats** |  | *Busqueda d'hipeparametres* |  |  |  
| [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) | set 2 i set 3 | trees: 20, criterion: gini, max depth: none, max features: log2 | 99.62% | 17.96s |  
| Custom Classifier | set 2 i set 3 | Boosting of: Logistic Regressor: {penalty: l1, C: 10} + Random Forest | 99.64% | 19.16s |

## Demo
Per tal de fer una prova, es pot fer servir amb la següent comanda:
``` python3 demo.py ```  
A continuació, després de que es generin els dos models, es poden entrar contrasenyes per veure els valors de seguretat que se li assignaria.

Exemple de la demo  
![demo_example](https://user-images.githubusercontent.com/57794658/148819093-35148cf4-4762-44da-89f8-a060cfa6a1ca.png)

## Conclusions
El millor model que s'ha aconseguit ha estat el doble if a partir de la llargada ja que aconsegueix el 100%.
L'altre molt bon model ha estat el Custom Classifier amb una precisio de mitja entre les classes de 99.637%. Aquest segon model consisteix en: primer deduir quan son del segon nivell de seguretat amb el primer classificador ([Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)), segon separar entre nivell zero i u de seguretat en els casos que la contrasenya no era de nivell dos amb els segon classificador. ([Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html))
## Idees per treballar en un futur
Crec que seria interesant refer el dataset però en comptes de treure les dades on els algoritmes no coincideixen, utilitzar la mitja de les seves qualificacions. 
## Llicencia
El projecte s’ha desenvolupat sota llicència MIT.
