import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Autor: Samuel Amico Fidelis
Versão: Final

"""

df = pd.read_csv('https://raw.githubusercontent.com/ect-info/ml/master/dados/DataBaseLop.csv')

### Dividir os alunos nas categorias de aprovados e reprovados

situacao = df["situacao"]
indx_apro = []
indx_repro = []
for i in range(len(situacao)):
    if(situacao[i] == 0):
        indx_repro.append(i)
    else:
        indx_apro.append(i)

print(("Total de alunos: {} , Aprovados = {} , Reprovados = {}").format(len(situacao),len(indx_repro),len(indx_apro)))


### Criando o Dataframe para alunos aprovados e reprovados:

df_aprovados = df.loc[indx_apro]
df_aprovados = df_aprovados.loc[:,['qsub1','qsub2','qsub3','qsub4','qsub5','qsubp1','qsubp2','totalsub','igualACeml123','igualACeml45','qsemana1','qsemana2','qsemana3','qsemana4','qsemana5','qsemana6','qsemana7','qsemana8','submeteu1','submeteu2','submeteu3','submeteu4','submeteu5','submeteu6','submeteu7','submeteu8','manha','tarde','noite','subListaLab23','subListaLab45','subListaLab67','subListaLab89','subListaLab1011','subListaExer23','subListaExer45','subListaExer67','subListaExer89','subListaExer1011','subDistintasLab23','subDistintasLab45','subDistintasLab67','subDistintasLab89','subDistintasLab1011','subDistintasExer23','subDistintasExer45','subDistintasExer67','subDistintasExer89','subDistintasExer1011','diferentesLabSemanas23','diferentesLabSemanas45','diferentesLabSemanas67','diferentesLabSemanas89','diferentesLabSemanas1011','diferentesExerSemanas23','diferentesExerSemanas45','diferentesExerSemanas67','diferentesExerSemanas89','diferentesExerSemanas1011'] ]

df_reprovados = df.loc[indx_repro]
df_reprovados = df_reprovados.loc[:,['qsub1','qsub2','qsub3','qsub4','qsub5','qsubp1','qsubp2','totalsub','igualACeml123','igualACeml45','qsemana1','qsemana2','qsemana3','qsemana4','qsemana5','qsemana6','qsemana7','qsemana8','submeteu1','submeteu2','submeteu3','submeteu4','submeteu5','submeteu6','submeteu7','submeteu8','manha','tarde','noite','subListaLab23','subListaLab45','subListaLab67','subListaLab89','subListaLab1011','subListaExer23','subListaExer45','subListaExer67','subListaExer89','subListaExer1011','subDistintasLab23','subDistintasLab45','subDistintasLab67','subDistintasLab89','subDistintasLab1011','subDistintasExer23','subDistintasExer45','subDistintasExer67','subDistintasExer89','subDistintasExer1011','diferentesLabSemanas23','diferentesLabSemanas45','diferentesLabSemanas67','diferentesLabSemanas89','diferentesLabSemanas1011','diferentesExerSemanas23','diferentesExerSemanas45','diferentesExerSemanas67','diferentesExerSemanas89','diferentesExerSemanas1011'] ]

### As analises pode ser feitas atraves destes dataframes,
### acesse o arquivo jupyter para mais detalhes


# MACHINE LEARNING - NEURAL NETWORK

X = df.loc[:,['qsub1','qsub2','qsub3','igualACeml123','submeteu1','submeteu2','submeteu3','subListaExer23','subListaLab23'] ]
y = df.loc[:,'situacao']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Treinando a parte de encoder
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

clf = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,10,10,9,1), random_state=1)

clf.fit(X_train, y_train) 
print(clf)

y_pred = clf.predict(X_test)

#print(y_test)
y_pred = (y_pred > 0.5)
#print(y_pred)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("Matriz de Confusão:")
print(cm)
print("Taxa de acerto:")
print((cm[0,0]+cm[1,1])/len(y_test) )
print(len(y_test))