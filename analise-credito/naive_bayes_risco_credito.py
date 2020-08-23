# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 12:35:32 2020

@author: renan
"""

import pandas as pd

arquivo = 'https://raw.githubusercontent.com/iGuntherr/machine-learning/master/analise-credito/risco_credito.csv'
base = pd.read_csv(arquivo)
previsores = base.iloc[:,0:4].values
classe = base.iloc[:, -1]

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder.fit_transform(previsores[:, 1])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])

from sklearn.naive_bayes import GaussianNB 
classificador = GaussianNB()
classificador.fit(previsores, classe)

# Historia boa, divida alta, garantias nenhuma, renda > 35
# Historia ruim, divida alta, garantias adequada, renda < 15 - Correcao laplaciana
resultado = classificador.predict([[0,0,1,2], [3,0,0,0]])

print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)