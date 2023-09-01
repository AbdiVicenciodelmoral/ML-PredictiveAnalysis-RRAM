#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

def synthSetVoltage(df):
        voltage = np.arange(0, 10.1, 0.1).tolist()
        rows = df.values
        idx = df.index.values
        new_rows = []
        for i,r in zip(idx,rows):
            r = r.tolist()
            r.append(i)
            for v in voltage:
                new_r = r.copy()
                new_r.append(v)
                new_rows.append(new_r)
                
        df = pd.DataFrame(new_rows, columns =['Baking temperature (°C)','Baking time (hours)','Resistive switching film',
                                     'Bottom electrode material','Top electrode material','Sample','SET Voltage']) 
        df.set_index('Sample', inplace=True)
        return df

def calcProb(s,kde):
    return np.exp(kde.score([[s]]))


def getKDE(idx,df):
    kdes = []
    synth_kdes = []
    # instantiate kde
    u = np.linspace(-1,14,500)
    # The grid we'll use for plotting
    x_grid = np.linspace(-1, 14, 1000)
    kde_dicts = {}
    for i in idx:
        d = df.loc[i]
        synth_d = df.loc[i].head(1)
        synth_d = synth_d.drop('SET Voltage',axis=1)
        synth_d = synthSetVoltage(synth_d)
        setVals = d['SET Voltage'].to_numpy()
        x = setVals.reshape(-1,1)
        sh = x.shape[0]
        grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': 10 ** np.linspace(-1, 1, 100)},cv=20) 
        grid.fit(x)
        kde = grid.best_estimator_
        kde_dicts[i] = kde
        v = d['SET Voltage'].apply(lambda x: calcProb(x,kde)).values
        synth_v =  synth_d['SET Voltage'].apply(lambda x: calcProb(x,kde)).values
        d = d.assign(Set_voltage_kde=v)
        synth_d = synth_d.assign(Set_voltage_kde=synth_v)
        #display(synth_d)
        kdes.append(d)
        synth_kdes.append(synth_d)
        
    return kdes,synth_kdes

