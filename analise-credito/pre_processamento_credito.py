import pandas as pd 
import numpy as np
arquivo = 'https://raw.githubusercontent.com/iGuntherr/machine-learning/master/analise-credito/credit_data.csv'
base = pd.read_csv(arquivo)
base.describe()

base.loc[base['age'] < 0]
# Apagar a coluna
#base.drop('age', 1, inplace=True)

# Apagar somente os registros com problema
#base.drop(base[base.age < 0].index, inplace=True)

# Preencher os valores com a media
base.mean() # Media da base completa
base['age'].mean() # Trazendo a media de todos os valores da coluna age (Idade) -- 40.80755937840458
base['age'][base.age > 0].mean() # Trazendo todos os valores com idade maior que 0 -- 40.92770044906149
base.loc[base.age < 0, 'age'] = 40.92 # Substituindo idades negativas

pd.isnull(base.age) # Mostra valores das bases como True para dados nulos e False para os preenchidos
base.loc[pd.isnull(base['age'])] # Usando o comando loc para trazer os valores nulos (True)

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer
# Configurando qual e meu campo nulo e qual a estrat'egia devera seguir para sugstituir os dados
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
# Fazendo a aplicacao da estrategia na variavel previsores, mas mantendo os resultados dentro 
# da variavel imputer
imputer = imputer.fit(previsores[:, 0:3])
# Aplicando a substituicao dentro da variavel previsores
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
