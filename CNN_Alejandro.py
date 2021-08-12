# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 15:53:56 2021

@author: Usuario
"""

# Numpy
import numpy as np 
# Matplotlib
import matplotlib.pyplot as plt
# Pandas
import pandas as pd 
from pandas import datetime

from sklearn.metrics import mean_squared_error
# Time
import time
# Yfinance
import yfinance as yf 

import os

import random

import tensorflow as tf

from pprint import pprint

pd.set_option('precision', 4)
np.set_printoptions(suppress=True, precision=4)
os.environ['PYTHONHASHSEED'] = '0'

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense

## CARGA DE DATOS EN PYTHON

def yf_data_ETL_7d1m(active_name): 
    
    '''
    Le pasas el nombre del activo y extrae un dataset con los valores del mismo para cada minuto durante 
    los últimos siete días. Ejemplos: 'EURUSD=X', 'BTC-USD', etc. Luego divide ese dataset en uno de 
    entrenamiento y otro testeo.
    '''
    
    data = yf.download(tickers=active_name, period='7d', interval='1m')
    df = pd.DataFrame(data=data)
    train_data, test_data = df[0:int(len(df)*0.8)], df[int(len(df)*0.8):]
    
    return df, train_data, test_data

df, train_data, test_data = yf_data_ETL_7d1m('BTC-USD')

data = df['Close'] # Solo vamos a trabajar con la columna "Close", no nos hacen falta todas.

data = (data - data.mean())/data.std() # Normalización gaussiana de los datos.

p = data.values

p = p.reshape((len(p), -1))

print(p)

## RED NEURONAL

lags = 5

g = TimeseriesGenerator(p, p, length=lags, batch_size = 5)

def create_rnn_model(hu = 100, lags = lags, layer = 'SimpleRNN', features = 1, algorithm = 'estimation'):
    model = Sequential()
    if layer == 'SimpleRNN':
        model.add(SimpleRNN(hu, activation = 'relu', input_shape = (lags, features)))
    else:
        model.add(LSTM(hu, activation = 'relu', input_shape = (lags, features)))
    if algorithm  == 'estimation':
        model.add(Dense(1, activation = 'linear'))
        model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
    else: 
        model.add(Dense(1, activation = 'sigmoid'))
        model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
    return model 

model = create_rnn_model()
# %time 
model.fit(g, epochs = 500, steps_per_epoch=10, verbose = False)

y = model.predict(g, verbose = False)

data = pd.DataFrame(data)
data['pred'] = np.nan
data['pred'].iloc[lags:] = y.flatten()

data[['Close', 'pred']].plot(
figsize = (10,6), style = ['b', 'r.'], alpha = 0.75)

plt.show()

data[['Close', 'pred']].iloc[50:100].plot(
figsize = (10,6), style = ['b', 'r-.'], alpha = 0.75)

plt.show()




