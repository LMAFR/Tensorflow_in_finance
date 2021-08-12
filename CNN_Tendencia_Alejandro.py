# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 15:53:56 2021

@author: Usuario
"""

## LIBRERÍAS

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

from sklearn.metrics import accuracy_score

## CARGA DE DATOS EN PYTHON

# Si se desea cargar datos actuales se puede usar la siguiente función:

# def yf_data_ETL_7d1m(active_name): 
    
#     '''
#     Le pasas el nombre del activo y extrae un dataset con los valores del mismo para cada minuto durante 
#     los últimos siete días. Ejemplos: 'EURUSD=X', 'BTC-USD', etc. Luego divide ese dataset en uno de 
#     entrenamiento y otro testeo.
#     '''
    
#     data = yf.download(tickers=active_name, period='7d', interval='1m')
#     df = pd.DataFrame(data=data)
#     train_data, test_data = df[0:int(len(df)*0.8)], df[int(len(df)*0.8):]
    
#     return df, train_data, test_data

# df, train_data, test_data = yf_data_ETL_7d1m('BTC-USD')

# Aquí vamos a usar datos fijos para poder ir comparando los resultados si fuera necesario:

df = pd.read_csv(r'C:\Users\Usuario\Desktop\Data Science\Trabajo\datos_bitcoin.csv')

## SELECCIÓN DE COLUMNAS Y CREACIÓN DE COLUMNAS CON OTRAS VARIABLES FINANCIERAS

data = df['Close'] # Solo vamos a trabajar con la columna "Close", no nos hacen falta todas.

data = pd.DataFrame(data)
window = 20

data['mide_tend'] = np.log(data['Close']/data['Close'].shift(1))
data['mom'] = data['mide_tend'].rolling(window).mean() # Variable "momento"
data['vol'] = data['mide_tend'].rolling(window).std() # Variable "volatilidad" (volumen)

data.dropna(inplace = True)

## CREACION DE LOS CONJUNTOS DE ENTRENAMIENTO Y TESTEO

split = int(len(data)*0.8)
train = data[:split].copy()
test = data[split:].copy()

## NORMALIZACIÓN GAUSSIANA

mu, std = train.mean(), train.std()

train = (train - mu)/std
test = (test - mu)/std

## RED NEURONAL Y SEMILLAS

lags = 5

# Función con nuestro modelo de red neuronal

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

# Función para establecer una semilla (para que los resultados sean reproducibles)

def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds()

## CREACIÓN Y AJUSTE DEL MODELO

model = create_rnn_model(features = len(data.columns))

g = TimeseriesGenerator(train.values, train['mide_tend'].values, length=lags, batch_size = 5)

model.fit(g, epochs = 100, steps_per_epoch=10, verbose = False)

## PREDICCIÓN Y TESTEO

g_ = TimeseriesGenerator(test.values, test['mide_tend'].values, length = lags, batch_size=5)

predictions = model.predict(g_).flatten() # El flatten es para que el array devuelto sea horizontal.

print(accuracy_score(np.sign(test['mide_tend'].iloc[lags:]), np.sign(predictions))) # 53.18%.

# Tengo que hacer la comparación con el caso sin normalización y extra de variables, ya que cuando lo hice aún usaba
# un dataset variable. Sin embargo, el resultado no ha variado significativamente en ninguna ejecución
# y un modelo normalizado ayuda a que todas las variables tengan un peso similar (esto no ha influido ahora,
# pero podría hacerlo en un futuro).

## REPRESENTACIÓN GRÁFICA

test['pred'] = np.nan
test['pred'].iloc[lags:] = predictions
test.dropna(inplace = True)

# En el siguiente gráfico no se aprecia bien la comparación entre la predicción y los datos de testeo:

test[['mide_tend','pred']].plot(figsize=(10,6), style = ['b','r-.'], alpha = 0.75);
plt.axhline(0, c='grey', ls = '--')
plt.show()

# En este otro se puede observar con más detalle un ejemplo de cómo se comportan las predicciones:

test[['mide_tend','pred']].iloc[1780:].plot(figsize=(10,6), style = ['b','r-.'], alpha = 0.75);
plt.axhline(0, c='grey', ls = '--')
plt.show()

# Con esto hemos convertido un problema de predicción en uno de clasificación (aunque hemos perdido información en el 
# proceso, en el sentido de que también es importante conocer cuánto varían los precios, no solo hacia dónde varían).
