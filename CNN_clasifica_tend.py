# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 10:04:38 2021

En los scripts previos he aplicado la redes neuronales para predecir el precio del bitcoin con respecto al dólar o he
convertido el problema al final en un problema de clasificación y he intentado predecir la tendencia en el precio. Sin
embargo, en este último caso no habíamos planteado el problema como uno de clasificación desde el primer momento, sino
que planteábamos un problema de predicción y en última instancia aprovechábamos los resultados para clasificar los 
resultados según si el precio subía o bajaba.

En este script vamos a plantear, desde el principio, un problema de clasificación para predecir dicha tendencia del precio.
Por último, comentar que vamos a usar unos pesos específicos para paliar el posible desbalanceo de clases (que haya muchos
valores que indiquen subida o bajada y eso pueda decantar el comportamiento de la red neuronal hacia uno de los dos lados de
forma absoluta o predominante) y vamos a usar una red neuronal del tipo LSTM, que se caracteriza por tener una capa de 
clasificación al final (en la parte de Short-Term), lo que la hace más útil que una RNN simple para problemas de 
clasificación. 

El resultado obtenido ha sido un 53.29% de acierto sobre el conjunto de datos utilizado con un conjunto de
entrenamiento que constituía un 80% del total del conjunto (proporción 80-20).

Por otro lado, he decidido implementar dropouts para evitar el sobreajuste del modelo.
Al hacerlo, aprovecho para crear una red con 2 capas ocultas, lo que también es un añadido adicional al análisis.

Los dropouts proporcionan cierta probabilidad a la red de encontrarse nodos deshabilitados en cada evaluación (esto emula
la perdida de información que sufre el cerebro humano mediante el olvido, dando lugar a nuevas posibilidades gracias a 
ignorar nodos de forma más o menos aleatoria).

No he implementado regularización porque sé cómo hacerlo para capas Dense, pero no se si se puede para las que uso aquí+
(el parámetro a emplear en capas Dense no aparece en las capas usadas aquí).

QUEDA PENDIENTE DE REVISAR QUE LOS RESULTADOS OBTENIDOS SEAN TODOS CON LA SEMILLA (SEED) REINICIADA.

@author: Alejandro Florido
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

from keras.layers import Dropout

from keras.regularizers import l1, l2

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

## CREACIÓN DE UNA VARIABLE BINARIA PARA CLASIFICAR LA TENDENCIA 

train_y = np.where(train['mide_tend']>0, 1, 0)
test_y = np.where(test['mide_tend'] > 0, 1, 0)

## DEFINICIÓN DE LOS PESOS PARA EVITAR EL DESBALANCEO DE CLASES EN VARIABLES DISCRETAS

def cw(a):
    c0, c1 = np.bincount(a)
    w0 = (1/c0)*(len(a))/2
    w1 = (1/c1)*(len(a))/2
    return {0: w0, 1:w1}

## RED NEURONAL Y SEMILLAS

lags = 5

# Función con nuestro modelo de red neuronal

# Modelo simple: (sin varias hl ni dropouts)

# def create_rnn_model(hu = 100, lags = lags, layer = 'SimpleRNN', features = 1, algorithm = 'estimation'):
#     model = Sequential()
#     if layer == 'SimpleRNN':
#         model.add(SimpleRNN(hu, activation = 'relu', input_shape = (lags, features)))
#     else:
#         model.add(LSTM(hu, activation = 'relu', input_shape = (lags, features)))
#     if algorithm  == 'estimation':
#         model.add(Dense(1, activation = 'linear'))
#         model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
#     else: 
#         model.add(Dense(1, activation = 'sigmoid'))
#         model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
#     return model

# Modelo profundo con Dropouts:

def create_deep_rnn_model(hl = 2, hu = 100, lags = lags, layer = 'SimpleRNN', features = 1,
                     optimizer = 'rmsprop', dropout = False, rate = 0.3, seed = 100
                     ):
    
    if hl <= 2:
        hl = 2
    if layer == 'SimpleRNN':
        layer = SimpleRNN
    else: 
        layer = LSTM
    
    model = Sequential()
    
    model.add(layer(hu, input_shape = (lags, features), return_sequences= True))
    if dropout:
        model.add(Dropout(rate, seed = seed))
        
    for _ in range(2, hl):
        model.add(layer(hu, return_sequences=True))
        
        if dropout:
            model.add(Dropout(rate, seed = seed))
    
    model.add(layer(hu))
    model.add(Dense(1, activation = 'sigmoid'))
        
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        
    return model

# Función para establecer una semilla (para que los resultados sean reproducibles)

def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds()

## CREACIÓN Y AJUSTE DEL MODELO

model = create_deep_rnn_model(hl = 2, hu = 50, layer = 'LSTM', features = len(data.columns), dropout = True, rate = 0.3)

g = TimeseriesGenerator(train.values, train_y, length = lags, batch_size= 5)

model.fit(g, epochs = 100, steps_per_epoch=10, verbose = False, class_weight=cw(train_y))

## PREDICCIONES

g_ = TimeseriesGenerator(test.values, test_y, length = lags, batch_size = 5)

predictions = model.predict(g_)

predictions_bool = np.where(predictions > 0.5, 1, 0)

print(accuracy_score(test_y[lags:], predictions_bool)) # 53.29% de acierto sin Dropouts y LSTM normal
                                                # 52.87% de acierto con Dropouts y LSTM deep
                                                # 51.25% de acierto con Dropouts y SimpleRNN deep
                                                
## REPRESENTACIÓN GRÁFICA

test_pred = pd.DataFrame(test_y, columns = ['test_data']).copy()
test_pred['bool_pred'] = np.nan
test_pred['bool_pred'].iloc[lags:] = predictions_bool.flatten()
test_pred.dropna(inplace = True)

# En el siguiente gráfico no se aprecia nada la comparación entre la predicción y los datos de testeo:

test_pred[['test_data','bool_pred']].plot(figsize=(10,6), style = ['b','r-.'], alpha = 0.75);
plt.axhline(0, c='grey', ls = '--')
plt.show()

# En este otro se puede observar mejor un ejemplo de cómo se comportan las predicciones:

test_pred[['test_data','bool_pred']].iloc[1870:].plot(figsize=(10,6), style = ['b','r-.'], alpha = 0.75);
plt.axhline(0, c='grey', ls = '--')
plt.show()

# Con esto hemos convertido un problema de predicción en uno de clasificación (aunque hemos perdido información en el 
# proceso, en el sentido de que también es importante conocer cuánto varían los precios, no solo hacia dónde varían).
