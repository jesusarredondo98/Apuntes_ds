# %% [markdown]
# # Dependencias

# %%
import numpy as np
import pandas as pd
import missingno as msgno

import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf

from datetime import date,time,datetime
from dateutil.relativedelta import relativedelta as rd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import MDS,TSNE
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.cluster import AgglomerativeClustering,KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score,mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import ks_2samp
from scipy.stats import kruskal
from statsmodels.stats.multicomp import MultiComparison




from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV

from varclushi import VarClusHi
from scipy import stats

from functools import reduce
from itertools import combinations

import json 
from glob import glob 
import os

from multiprocessing import Pool
from multiprocessing import get_context


from warnings import filterwarnings

pd.set_option('display.max_columns',None)
pd.set_option('display.float_format',lambda x:'%.2f'%x)
filterwarnings('ignore')
cf.go_offline()

# %% [markdown]
# # Entendimiento de la data

# %%
df = pd.read_csv("../Examen/Data/hotel_bookings.csv")
df.head(5)

# %% [markdown]
# # Ingeniería de caracteristicas

# %%
df.insert(0,"Id Transaccion",df.index+1)
df.head()


# %% [markdown]
# Entendimiento del problema:
# 
# Buscamos:
# 
# Utilizando los datos adjuntos genere un resumen ejecutivo de sus recomendaciones técnicas respaldadas por analítica para atender lo siguiente:
# 
# - Disminución de la tasa de cancelación.
# - Maximizar el número de clientes recurrentes.
# - Maximizar la longitud de la estancia.
# 
# Lo anterior podria tratarse de un problema de clusterizacion para determinar los problemas de maximizacion de numero de clientes recurrentes y la longitud de la estancia y para la disminucion de la tasa de cancelacion podría tratarse de un problema de clasificación ocupando un modelo explicativo como una regresion logistica o un arbol de decisión.

# %%
#Definicion de variables
df.columns

# %%
df.info()

# %%
# Para un problema de clasificacipon
um = ["Id Transaccion"]
vart = ["is_canceled"]
vard = ["hotel","meal","country","market_segment","distribution_channel","is_repeated_guest","reserved_room_type","assigned_room_type","deposit_type","customer_type",\
       "reservation_status"]
varc = ["stays_in_weekend_nights","stays_in_week_nights","adults","children","babies","previous_cancellations","previous_bookings_not_canceled","booking_changes",\
       "days_in_waiting_list","adr","total_of_special_requests"]
vardates = ["reservation_status_date"]



# %%
"""Vemos cuantas columnas nos quedaron"""
df[um+vart+vard+varc].shape,df.shape

# %%
"""Veamos que todas las variables continuas sean numericas"""
df[varc].dtypes

# %%
df[varc].head(5)

# %%
"""Comprobamos que las variables continuas esten en un formato aceptable"""
df[varc].dtypes

# %%
#Creacion de nuevas variables
#Tasa de cancelaciones
df["cancelation_rate"] = df["previous_cancellations"]/(df["previous_cancellations"]+df["previous_bookings_not_canceled"])
#Numero noches totales de hospedaje
df["total_stays_nights"] = df["stays_in_week_nights"]+df["stays_in_weekend_nights"]

varc = varc + ["cancelation_rate","total_stays_nights"]
df.head(5)

# %%
#Variables continuas finales
varc

# %%
#Variables discretas
df[vard].dtypes

# %%
#Cambiamos is_repeated_guest a texto
df["is_repeated_guest"] = np.where(df["is_repeated_guest"]==1, "Yes", "No")
df.head(5)

# %% [markdown]
# # Modelación Supervisada
# 
# Objetivo: Disminucion tasa de cancelación

# %%
df_ = df[um+vart+varc+vard].copy()
df_.shape,df.shape

# %% [markdown]
# ## Particion

# %%
df_[vart].value_counts()

# %%
#Como la base esta desbalanceada trabajaremos solo con 40000 no cancelados y 40000 cancelados
noc,c = [d.reset_index(drop=True) for _,d in df_.groupby('is_canceled')]

# %%
rs = rs = np.random.RandomState(seed=20)
noc = noc.sample(n=40000,random_state=rs).reset_index(drop=True)
c = c.sample(n=40000,random_state=rs).reset_index(drop=True)
noc.shape,c.shape

# %%
#Juntamos los dos df
muestra = pd.concat([noc,c],ignore_index=True)

# %%
muestra[vart].value_counts(normalize=1)

# %%
muestra.head(5)

# %%
"""Dividimos el dataset en conjunto de entrenamiento y prueba"""
train,valid = train_test_split(muestra,train_size=0.7,random_state=rs)
train.reset_index(drop=True,inplace=True)
valid.reset_index(drop=True,inplace=True)
train.shape,valid.shape

# %% [markdown]
# ### EDA

# %%
#Para las fechas
df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"],format='%Y-%m-%d').dt.date
df.info()

# %%
df.head(5)

# %%
aux = df[["reservation_status_date","Id Transaccion"]].groupby(by="reservation_status_date").count().plot(figsize=(10,5))

# %%
aux = df[["reservation_status_date","Id Transaccion"]].groupby(by="reservation_status_date").count().sort_values(by = "Id Transaccion",ascending=False)
aux.head()


# %% [markdown]
# ### Continuos

# %%
"""Para la variables agrupadas como continuas las hacemos numericas"""
for v in varc:
    train[v] = pd.to_numeric(train[v],errors='coerce').replace({np.inf:np.nan,-np.inf:np.nan})

# %% [markdown]
# #### Ausentes

# %%
"""Creamos una tabla para identificar el numero de datos ausentes"""
miss = 1-train[varc].describe().T[['count']]/len(train)
miss.sort_values(by='count',ascending=False,inplace=True)
miss

# %%
import missingno as msno
msno.matrix(train[varc])

# %%
train["children"].value_counts(dropna=False)

# %%
def missing_zero_values_table(df):

        zero_val = (df == 0.00).astype(int).sum(axis=0)

        mis_val = df.isnull().sum()

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)

        mz_table = mz_table.rename(

        columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})

        mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']

        mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(df)

        mz_table['Data Type'] = df.dtypes

        mz_table = mz_table[

            mz_table.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      

            "There are " + str(mz_table.shape[0]) +

              " columns that have missing values.")

        return mz_table
    
missing_zero_values_table(train[varc])

# %%
# A cancelarion_rate las llenamos con 0 y con children usaremos un simple imputer con la mediana
train["cancelation_rate"] = train["cancelation_rate"].fillna(0)
im = SimpleImputer(strategy='median')
im.fit(train[varc])
Xi = pd.DataFrame(im.transform(train[varc]),columns=varc)
Xi.dropna().shape,Xi.shape

# %%
missing_zero_values_table(Xi[varc])

# %%
ks = pd.DataFrame(map(lambda v:(v,stats.ks_2samp(Xi[v],train[v]).statistic),varc),columns=['var','ks'])
ks

# %%
fuera = list(ks[ks['ks']>0.1]['var'])
len(fuera)

# %% [markdown]
# Ninguna variable se vio afectada

# %%
varc = [v for v in varc if v not in fuera]
train.drop(fuera,axis=1,inplace=True)

# %% [markdown]
# #### Varianza Explicada

# %%
vt = VarianceThreshold()
vt.fit(Xi)

# %%
fuera = [v for v,var in zip(varc,vt.get_support()) if not var]
print(len(fuera))
varc = [v for v in varc if v not in fuera]

# %% [markdown]
# #### Multicolinealidad

# %%
"""Usando varclushi se generan cluster de variables por su colinealidad"""
vc = VarClusHi(df=Xi,feat_list=varc)
vc.varclus()

# %%
vc.rsquare.sort_values(by=['Cluster','RS_Ratio'])

# %%
best = list(vc.rsquare.sort_values(by=['Cluster','RS_Ratio']).groupby('Cluster').first()['Variable'])
best

# %%
Xi = Xi[best].copy()
Xi.head(5)

# %% [markdown]
# #### Extremos

# %%
"""Metodo multivariante, declaramos el # componentes igual a 10 por su intepretabilidad pero puede rondear por ese numero"""
gmm = GaussianMixture(n_components=10)
gmm.fit(Xi[best]) #entrenamos
Xi['ex_mv'] = gmm.predict(Xi[best]) #predecimos
Xi['ex_mv'].value_counts(1) #checamos los elementos por clause y si es menor a 0.05 se dice que es outlier

# %%
#Eliminamos los grupos menores a 0.05
Xi.loc[~Xi['ex_mv'].isin([4,3,1,2,8,0])].reset_index(drop=True).drop(Xi.filter(like='ex_').columns,axis=1)[best].hist(figsize=(10,10))

# %%
"""Quitamos los outliers identificados por el metodo multivariado"""
print(Xi.shape[0])
Xi = Xi.loc[~Xi['ex_mv'].isin([4,3,1,2,8,0])].reset_index(drop=True).drop(Xi.filter(like='ex_').columns,axis=1)
print(Xi.shape[0])

# %%
54111/56000

# %%
Xi

# %% [markdown]
# #### Poder predictivo

# %%
"""Unimos la variable objetivo"""
Xi[um] = train[um]
Xi = Xi.merge(train[um+vart],on=um,how='inner')
Xi.head(5)

# %%
"""Aplicamos selectkbest a con numero de las variables seleccionadas por multicolinealidad"""
sk = SelectKBest(k=len(best))
"""Entrenamos con las variables escogidas por multicolinealidad con respecto a la target"""
sk.fit(Xi[best],Xi[vart[0]])

# %%
"""Visualizamos su poder predictivo"""
pd.DataFrame(zip(best,sk.scores_),
columns=['var','score']).sort_values(by='score',ascending=False).set_index('var').iplot(kind='bar',color='purple')

# %%
"""Removemos la de valor mas bajo"""
best.remove('cancelation_rate')
best.remove('previous_bookings_not_canceled')

# %%
best

# %%
Xi[best].hist() #vemos sus distribucioones

# %% [markdown]
# #### Consideraciones Finales

# %%
im.fit(train[best])

# %% [markdown]
# ### Discreto

# %%
"""Funcion para crear la tabla de frecuencias"""
def freq(df:pd.DataFrame,var:list):
    if type(var)!=list:
        var = [var]
    for v in var:
        aux = df[v].value_counts().to_frame().sort_index()
        aux.columns = ['FA']
        aux['FR'] = aux['FA']/aux['FA'].sum()
        aux[['FAA','FRA']] = aux.cumsum()
        print(f'****Tabla de frecuencias  {v}  ***\n\n')
        print(aux)
        print("\n"*3)

# %%
"""A los missings los mandamos a esa categoria"""
for v in vard:
    train[v] = train[v].fillna('MISSING')

# %%
freq(train,vard)

# %%
"""Funcion para normalizar las variables discetas"""
def normalizar(df:pd.DataFrame,var:str,umbral:float=0.05)->tuple:
    aux = df[var].value_counts(1).to_frame()
    aux['map'] = np.where(aux[var]<umbral,'Otros',aux.index)
    if aux.loc[aux['map']=='Otros'][var].sum()<umbral:
        aux['map'].replace({'Otros':aux.head(1)['map'].values[0]},inplace=True)
    aux.drop(var,axis=1,inplace=True)
    return var,aux['map'].to_dict()

mapa_norm = list(map(lambda v:normalizar(train,v),vard))

"""Se crean nuevas columnas ya con la normalizacion"""
for v,mapa in mapa_norm:
    train[f'n_{v}'] = train[v].map(mapa)

# %%
"""Lista de variables normalizadas"""
varn = [v for v in train.columns if v[:2]=='n_']

# %%
"""Variables con una categoria"""
unarias = [v for v in varn if train[v].nunique()==1]
unarias

# %%
"""Quitamos las unarias"""
varn = [v for v in varn if v not in unarias]
varn

# %% [markdown]
# ### Espacio vectorial
# 
# #### Espacio $\mathcal{X}$

# %%
"""Unimos las variables continuas y discretas eleginas usando como llave a um"""
X = Xi[um+best].copy().merge(train[um+varn],on=um,how='inner')
X.head()

# %%
"""Usamos OHE y se elimina una categoria si es binaria"""
oh = OneHotEncoder(drop='if_binary')

# %%
oh.fit(X[varn])
#Creamos las nuevas columnas con sus nombre signados
X[oh.get_feature_names_out()] = oh.transform(X[varn]).toarray()
#Eliminamos las variables anteriores, antes de los dummies
X.drop(varn,axis=1,inplace=True)

# %%
#Agrupamos las variables a modelas para modelar
var = best+list(oh.get_feature_names_out())

# %%
X.head()

# %%
X.shape

# %% [markdown]
# ### Aprendizaje Supervisado

# %%
"""Definimos el conjuunto de entrenamiento según la hiper caja elegida"""
Xt = X.copy()
Xt.shape

# %%
yt = Xt[um].merge(train[um+vart],on=um,how='inner')[vart[0]] #Guardamos la variable objetivo
Xt = Xt.drop(um,axis=1)

# %%
Xt.shape,yt.shape #comprobamos dimensiones

# %%
"""Instanciamos los modelos"""
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier()

# %% [markdown]
# #### Regresion Logistica

# %%
"""Cargamos los hiperparametros de la reg logistica"""
hp = {"penalty" : ['l1', 'l2', 'elasticnet', 'none'],
"C": np.arange(0,2,0.1)}

# %%
"""Realizamos una busqueda de gradilla para encontrar los mejores hiperparametros"""
grid = GridSearchCV(estimator=lr,param_grid=hp,n_jobs=-1,verbose=True,scoring='accuracy',cv=4)

# %%
grid.fit(Xt,yt) #entrenamos el modelo

# %%
lr = grid.best_estimator_ #usamos el modelo ganador

# %%
grid.best_score_ #vemos el score obtenido

# %%
grid.best_params_ #visualizamos los hiperparametros ganadores

# %%
lr.fit(Xt,yt) #Entrenamos el mejor modelo

# %%
lr.coef_,lr.intercept_ #visulizamos los parametros del modelo de reg log

# %%
Coeficientes=pd.DataFrame()
Coeficientes["Variable"]=list(Xt.columns)
Coeficientes["Coeficiente"]=lr.coef_.tolist()[0]
Coeficientes.sort_values(by="Coeficiente",ascending=False)

# %%
"""Vemos sus metricas"""
print(accuracy_score(y_true=yt,y_pred=lr.predict(Xt)))
print(roc_auc_score(y_true=yt,y_score=lr.predict_proba(Xt)[:,1]))

# %% [markdown]
# #### Generalizacion

# %%
"""Llenamos los valores faltantes como missing en la porcion de validacion de las variables discretas"""
for v in vard:
    valid[v] = valid[v].fillna('MISSING')

# %%
"""Normalizamos las variables discetras en la porcopm de validación"""
for v,mapa in mapa_norm:
    valid[f'n_{v}'] = valid[v].map(mapa)

# %%
for v in varn:
    valid[v] = valid[v].fillna('MISSING')

# %%
for v in varn:
    valid[v] = valid[v].map(lambda x: x.replace("MISSING","Otros"))


# %%
Xv[best].isnull().any()

# %%
"""Repetimos las transformaciones aplicadas a la porcion de train pero ahora a la de validación"""
Xv = pd.DataFrame(im.transform(valid[best]),columns=best)
Xv[um] = valid[um]
Xv = Xv[um+best].copy().merge(valid[um+varn],on=um,how='inner')
Xv[oh.get_feature_names_out()] = oh.transform(Xv[varn]).toarray()
Xv.drop(varn,axis=1,inplace=True)
#Xv = pd.DataFrame(sc01.transform(Xv[var]),columns=var)
#Xv = pd.DataFrame(pipe_pca.transform(Xv[var]))
#Xv.insert(0,um[0],valid[um])
yv = Xv[um].merge(valid[um+vart],on=um,how='inner')[vart[0]]
Xv = Xv.drop(um,axis=1)
Xv.shape,yv.shape #comprobamos dimensiones de la variables dependientes y la targed de la porcion de validación

# %%
Xv

# %%
"""Porcion de validacion"""
print(accuracy_score(y_true=yv,y_pred=lr.predict(Xv)))

# %%
"""Porcion de entrenamiento"""
print(accuracy_score(y_true=yt,y_pred=lr.predict(Xt)))

# %%
#Percerbamos el data set
import pickle
obj = (mapa_norm,im,oh,best,vart,varn,vard,um,var,train,valid)
pickle.dump(obj,open('objetos_persistencia.pkl','wb'))

# %%
espacios = (X,Xt,yt,Xv,yv)
pickle.dump(espacios,open('espacios.pkl','wb'))


# %% [markdown]
# # Modelación No Supervisada
# 
# Objetivos:
# - Maximizar el número de clientes recurrentes.
# - Maximizar la longitud de la estancia
# 
# ## Definicion de variables
# 
# 

# %%
varc

# %%
vard

# %%
vart

# %%
um

# %%
# Juntamos las variables discretas y la target
vard = vard + vart

# %% [markdown]
# ## EDA

# %%
df_c = df_[um+varc+vard].copy()
df_c

# %% [markdown]
# ### Continuas
# 
# #### Ausentes

# %%
"""Creamos una tabla para identificar el numero de datos ausentes"""
miss = 1-df_c[varc].describe().T[['count']]/len(df_c)
miss.sort_values(by='count',ascending=False,inplace=True)
miss

# %%
df_c["cancelation_rate"] = df_c["cancelation_rate"].fillna(0)
msno.matrix(df_c[varc])

# %%
missing_zero_values_table(train[varc])

# %%
"""Instanciamos el imputador bajo una estrategia en la mediana"""
im = SimpleImputer(strategy='median')
"""Con el imputador llenamos las variables con ausentes en un nuevo df"""
Xi = pd.DataFrame(im.fit_transform(df_c[varc]),columns=varc)
Xi.shape #Comprobamos dimensiones

# %%
# Comprobamos dimensiones sin ausentes y totales de xi para ver que no se elimino nongun registro
Xi.dropna().shape,Xi.shape

# %%
"""Comprobamos que no lastimamos la distribucion de ninguna variable"""
ks = pd.DataFrame(map(lambda v:(v,ks_2samp(Xi[v],df_c[v].dropna()).statistic),
                 varc),columns=['variable','ks'])

# %%
ks

# %%
"""Creamos una lista con las variables afectadas, aquellas con el estadistico mayor a 0.1 ó una p-value mayor a 0.05"""
rotas = ks.loc[ks['ks']>0.1]['variable'].tolist()
len(rotas)

# %% [markdown]
# #### Baja Varianza

# %%
"""Instanciamos Variance Threshold para ver que nuestras variables brinden algo de varianza"""
vt = VarianceThreshold(threshold=0.1)
vt.fit(Xi) #entrenamos con nuestras variables continuas ya imputadas
"""Agrupamos en una lista las variables que aportan baja variaza"""
varianza_peq = [v for v,nu in zip(varc,vt.get_support()) if not nu]
"""Nos quedamos con las variables continuas que si aporten variaza"""
varc = [v for v in varc if v not in varianza_peq] #variables continuas que aportan varianza
df_c.drop(varianza_peq,axis=1,inplace=True) #eliminamos en df_c
Xi.drop(varianza_peq,axis=1,inplace=True) #eliminamos en Xi que es el imputado

# %%
len(varc) #vemos con cuante nos quedamos

# %%
varianza_peq

# %%
varc

# %%
#Variables importantes total_stays_nights, 'previous_bookings_not_canceled, is_repeated_guest

# %% [markdown]
# #### Multicoleanidad

# %%
"""Usando varclushi se generan cluster de variables por su colinealidad"""
vc = VarClusHi(df=Xi,feat_list=varc)
vc.varclus()
"""Nos quedamos con aquellas variables de menor rs ratio de cada cluster"""
rs = vc.rsquare
rs = rs.sort_values(by=['Cluster','RS_Ratio']).reset_index(drop=True)
rs['id'] = rs.groupby('Cluster').cumcount()+1
mc = rs.loc[rs['id']==1]['Variable'].tolist()
mc

# %%
rs

# %%
#Cambiamos previous_cancelatio por previous_bookings_not_canceled dado que estn en el mismo cluster
mc = mc + ["previous_bookings_not_canceled"]
mc.remove("previous_cancellations")
mc

# %% [markdown]
# ## Discreto

# %%
"""Para todas las variables discretas llenamos los valores nulos como missing"""
for v in vard:
    df_c[v] = df_c[v].fillna('MISSING')

# %%
"""Aplicamos la función de fecuencias"""
freq(df,vard)

# %%
"""mapeamos las variables a normalizar"""
mapa_norm = list(map(lambda v:normalizar(df,v),vard))
"""Aplicamos para todas las variables discretas"""
for v,mapa in mapa_norm:
    df[f'n_{v}'] = df[v].map(mapa)
"""Creamos una lista de variables normalizadas"""
varn = [v for v in df.columns if v[:2]=='n_']
varn

# %%
"""Separamos a las variables disretas ya normalizada que son unarias"""
unarias = [v for v in varn if df[v].nunique()==1]
unarias

# %%
#Nos quedamos con is_repeated_guest
"""Eliminamos a las unarias"""
varn = [v for v in varn if v not in unarias]
varn = varn + ["is_repeated_guest"]
varn

# %%
"""Dataset de variables discretas normalizadas"""
Xd = df[varn].copy()
Xd[um] = df[um]
"""Tabla de frecuencia de las variables discretas ya normallizadas"""
freq(Xd,varn)

# %% [markdown]
# #### Warm Clustering

# %%
"""Ajustamos un cluster, cualquiera que se quiera."""
cl = GaussianMixture(n_components=5)
cl.fit(Xi[mc])

# %%
"""Se pone como variable objetivo"""
pd.Series(cl.predict(Xi[mc])).value_counts(1).plot(kind='pie')

# %%
"""Densidades de los grupos"""
pd.Series(cl.predict(Xi[mc])).value_counts(1)

# %%
mc

# %%
"""Seleccionamos cuales son las mejores variables"""
sk = SelectKBest(k=3) #seleccionamos el numero deseado
sk.fit(Xi[mc],cl.predict(Xi[mc]))

# %%
"""Creamos una lista con lo mejores """
best = [v for v,i in zip(mc,sk.get_support()) if i]
best #Como no se elimino una de las que consideramos importante nos quedamos con best para realizar el cluster

# %% [markdown]
# #### Cambio de Espacio

# %%
Xi[um] = df_c[um]


# %%
"""Muestra para visualizar los cluster, sirve como auxiliar"""
x = Xi.sample(n=500).reset_index(drop=True)

# %% [markdown]
# #### PCA $\mathcal{X}\to\mathcal{X}_p$

# %%
"""Creamos el espacio de componenetes principales"""
sc = StandardScaler()
pca = PCA(n_components=2) #Reducimos a dos dimensiones
Xp = pd.DataFrame(pca.fit_transform(sc.fit_transform(x[best]))) #creamos el data frame
print(pca.explained_variance_ratio_.cumsum()) #vemos la varianza explicada
Xp

# %% [markdown]
# #### MDS $\mathcal{X}\to\mathcal{X}_m$

# %%
"""Creamos el espacio de mds"""
sc = MinMaxScaler()
mds = MDS(n_components=2,n_jobs=-1) #con 2 dimensiones
Xm = pd.DataFrame(mds.fit_transform(sc.fit_transform(x[best]))) #creamos el df
Xm

# %% [markdown]
# #### t-SNE $\mathcal{X}\to\mathcal{X}_t$

# %%
"""Creamos el espacio de tsne"""
sc = MinMaxScaler()
tsne = TSNE(n_components=2,n_jobs=-1,perplexity=15) #con dos dimensiones, el hiperparametreo de perplexity es la clave para que quede mejor
Xt = pd.DataFrame(tsne.fit_transform(sc.fit_transform(x[best])))
Xt

# %% [markdown]
# ## Visualización preliminar

# %%
"""Grafica de PCA"""
Xp.iplot(kind='scatter',x=0,y=1,mode='markers',color='blue')

# %%
sns.kdeplot(data=Xp,x=0,y=1,fill=True)

# %%
"""Gráfica MDS"""
Xm.iplot(kind='scatter',x=0,y=1,mode='markers',color='blue')

# %%
sns.kdeplot(data=Xm,x=0,y=1,fill=True)

# %%
"""Gráfica TSNE"""
Xt.iplot(kind='scatter',x=0,y=1,mode='markers',color='purple')

# %%
sns.kdeplot(data=Xt,x=0,y=1,fill=True)

# %% [markdown]
# ## Clustering

# %%
sc = MinMaxScaler()
Xs = pd.DataFrame(sc.fit_transform(x[best]),columns=best)

# %% [markdown]
# ### K-Medias

# %%
sil = pd.DataFrame(map(lambda k:(k,silhouette_score(Xs,
                                              KMeans(n_clusters=k,max_iter=1000).fit_predict(Xs))),
                 range(2,10)),columns=['k','sil']).set_index('k')
sil.iplot(kind='line',mode='lines+markers',color='blue')

# %%
# Tomamos como ganador 2 clusters
k = 2
tipo = 'kme'
kme = KMeans(n_clusters=k,max_iter=1000)
x[f'cl_{tipo}']=Xp[f'cl_{tipo}']=Xm[f'cl_{tipo}']=Xt[f'cl_{tipo}'] =kme.fit_predict(Xs[best])

# %%
"""Seleccionamos el modelo que tenga mejor silueta global"""
varcl = sorted(x.filter(like='cl_'))
for v in varcl:
    Xp[v] = Xp[v].astype(str)
    Xm[v] = Xm[v].astype(str)
    Xt[v] = Xt[v].astype(str)
    x[v] = x[v].astype(str)
    
pd.DataFrame(map(lambda cl:(cl,silhouette_score(Xs,x[cl])),varcl),columns=['cluster','sil']).iplot(kind='bar',categories='cluster')

# %% [markdown]
# ## Visualizador con cluster

# %%
mejor = 'kme'
Xp.iplot(kind='scatter',x=0,y=1,mode='markers',categories=f'cl_{mejor}')
sns.kdeplot(data=Xp,x=0,y=1,fill=True,hue=f'cl_{mejor}')

# %%
Xm.iplot(kind='scatter',x=0,y=1,mode='markers',categories=f'cl_{mejor}')
sns.kdeplot(data=Xm,x=0,y=1,fill=True,hue=f'cl_{mejor}')

# %%
Xt.iplot(kind='scatter',x=0,y=1,mode='markers',categories=f'cl_{mejor}')
sns.kdeplot(data=Xt,x=0,y=1,fill=True,hue=f'cl_{mejor}')

# %%
x["cl_kme"].value_counts(normalize=1)

# %% [markdown]
# ## Perfilamiento

# %%
pd.DataFrame(map(lambda v:(v,
              kruskal(*[d[v].reset_index(drop=True) for cl,d in x[[f'cl_{mejor}',v]].groupby(f'cl_{mejor}')]).pvalue),best),
             columns=['variable','p-value']).round(2)

# %%
#Podemos apreciar que total_stays_nightd es una buena variable para perfilar

# %%
for v in best:
    print(v)
    display(MultiComparison(x[v],x['cl_kme']).tukeyhsd().summary())
    plt.figure()
    sns.boxplot(data=x,y=v,x='cl_kme')

# %% [markdown]
# #### Discreto

# %%



