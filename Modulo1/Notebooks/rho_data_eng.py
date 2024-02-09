import pandas as pd
import numpy as np
import missingno as msno 
import cufflinks as cf
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
pd.set_option('display.max_columns',100)

import plotly
import chart_studio.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.express as px


from plotly.graph_objs import Scatter, Figure, Layout
from plotly import tools
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from IPython.display import display
import datetime as dt

def missings_continous(df):
    
    ## return the missings values in the continous variables in ascending order
    
    miss=(df.isnull().sum()/df.shape[0]).to_frame('Missings').sort_values(by='Missings',ascending=False)
    return miss


def missings_discrets(df,list_v):
    
    ## return the missings values in the discretes variables
    ## distributes the missings values in the rest of the categories uniformly
    
    for var in list_v:
        aux=df[var].copy()
        nulos=aux[aux.isnull()]
        no_nulos=aux[~aux.isnull()]
        dict_=no_nulos.value_counts(normalize=True).to_dict()
        values=list(dict_.values())
        keys=list(dict_.keys())
        aux=aux.map(lambda x: list(np.random.choice(keys,1, p=values))[0] if x!=x else x)
        aux.value_counts(normalize=True)
        df[var]=aux.copy()          
            

def descripcion_variables(df,list_variables):
    
    ## return a dataframe with the description of the variables on the list with the acum. percentage for a better classification
    
    for var in list_variables:
        aux=df[var].copy()
        aux=df[var].value_counts(dropna=False).to_frame()
        aux['pct']=df[var].value_counts(normalize=True,dropna=False)
        aux['pct_acum']=aux['pct'].cumsum()
        display(aux)
        
        
def describe_(df, variables):
    
    return(df[variables].describe(percentiles = [.01, .05, .1, .25, .5, .75, .9, .95, .99]))


def histograms(df,list_v,frac=.3):
    
    #returns the histograms of the variables 
    
    for var in list_v:
        fig = px.histogram(df.sample(frac=frac), x=var,color_discrete_sequence= px.colors.sequential.Peach,
                          title=f'Hist of {var}',)
        fig.show()
        


def percentiles(data,l):
    for i in l:
        print(i)
        qn=data[i].quantile(.01)
        qm=data[i].quantile(.99)
        iqr=qm-qn
        data[f'per_{i}']=(data[i]<qn) | (data[i]>qm)
        data['dato_atipico']=data[[x for x in data.columns if 'per_' in x]].sum(axis=1)



def woe_iv_cat(df,target,list_var):
            
    ##Return the WOE and the IV of the variables 
            
    df[list_var].fillna('Missings',inplace=True)
    iv={}
    for var in list_var:
        tabla_y = df[[target,var]].pivot_table(index=var,columns=target,aggfunc='size')
        WOE = tabla_y.apply(lambda x: x/sum(x)).apply(lambda x: np.log(x[1]/x[0]),axis=1)
        tabla_y['woe'] = WOE
        tabla_y = tabla_y.apply(lambda x: x/sum(x))
        tabla_y['IV'] = list(map(lambda evento, no_evento, woe: (no_evento-evento)*woe,tabla_y[1],tabla_y[0],tabla_y['woe']))
        df = df.merge(tabla_y,left_on=var,right_index=True)
        IV = tabla_y['IV'].sum()
        display(tabla_y)
        iv.update({var:IV})
    return(iv)
            
def ks_hipo(tr, ts, var):
    
    from scipy.stats import ks_2samp 
    
    #Tomamos el data frame, y de ahí extraemos los cotrastes de hipótesis para el ks
    #tr: el dataframe de train
    #ts: el dataframe de test
    
    df = pd.DataFrame(columns=['variable', 'p-value', '¿Son iguales?'])
    
    for v in var:
        df.loc[v,'p-value'] = ks_2samp(tr[v], ts[v]).pvalue
        
    df["¿Son iguales?"] = df["p-value"].map(lambda x: "No" if x<0.05 else "Si")
    df['variable'] = df.index
    df.index = [i for i in range(len(df['variable']))]
    
    
    return(df)

def chi_hipo(tr, ts, var):
    
    from scipy.stats import chisquare
    
    #Tomamos el data frame, y de ahí extraemos los cotrastes de hipótesis para el ks
    #tr: el dataframe de train
    #ts: el dataframe de test
    
    df = pd.DataFrame(columns=['variable', 'p-value', '¿Son iguales?'])
    
    for x in var:
        df.loc[x, "p-value"] = chisquare(f_obs=ts[x].value_counts(True).sort_index().values, f_exp=tr[x].value_counts(True).sort_index().values).pvalue
        
    df["¿Son iguales?"] = df["p-value"].map(lambda x: "No" if x<0.05 else "Si")
    df['variable'] = df.index
    df.index = [i for i in range(len(df['variable']))]
    
    return(df)    

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
    
from math import radians, cos, sin, asin, sqrt
import pandas as pd

def lat_long_to_distance(latitudes_1, latitudes_2, longitudes_1, longitudes_2,
                        units = 'Kilometers'):
    
    '''This function obtains the distance between two Earth points using Haversine
        formula it has as inputs the series of data.
        
        Args:
        1. latitudes_1: an array containing the start latitudes.
        2. latitudes_2: an array containing the end latitudes.
        3. longitudes_1: an array containing the start longitudes.
        4. longitudes_2: an array containing the end longitudes.
        5. units: the unit for the result. Possible values 'Kilometers' 'Miles'
        
        Returns:
        1. A pandas series with the distance in kilometers.'''
    
    la1 = len(latitudes_1)
    la2 = len(latitudes_2)
    lg1 = len(longitudes_1)
    lg2 = len(longitudes_2)
    
    if (la1 == la2 == lg1 == lg2):
        
        #Radians conversion:
        lat1 = pd.Series(latitudes_1).map(radians)
        lat2 = pd.Series(latitudes_2).map(radians)
        lon1 = pd.Series(longitudes_1).map(radians)
        lon2 = pd.Series(longitudes_2).map(radians)
        
        #Haversine:
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        #a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        a = (dlat.map(lambda x:sin(x/2)**2) + lat1.map(cos) * lat2.map(cos) * 
             dlon.map(lambda x:sin(x/2)**2))
        
        #c = 2 * asin(sqrt(a))
        c = a.map(sqrt).map(asin) * 2
        
        #Earth Radius:
        if units == 'Kilometers':
            r = 6371
        elif units == 'Miles':
            r = 3956
        else: return('Invalid units.')
        
        return(c*r)
        
    else:
        
        print(la1, la2, lg1, lg2)
        return('Invalid dimensions.')

def iqr_(df, variables, alpha = 1):

    
    for v in variables:
        q3 = df[v].quantile(.75)
        q1 = df[v].quantile(.25)        
        iqr = q3 - q1
        lb, up = q1-(alpha*iqr), q3+(alpha*iqr)
        df = df.loc[(df[v]>=lb) & (df[v]<=up)].copy()
        
    return(df)    
