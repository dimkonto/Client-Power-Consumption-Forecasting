# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 10:52:46 2021

@author: dimkonto
"""

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
from sklearn import preprocessing
import scipy
from tabulate import tabulate
from matplotlib import pyplot as pp
import datetime
import statistics
import math
from scipy import stats

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tslearn.utils import to_time_series
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import soft_dtw

import seaborn as sns
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from keras.callbacks import TensorBoard

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
import statsmodels.formula.api as smf
from itertools import product
from scipy.optimize import differential_evolution


#PER CLIENT CONSUMPTION
path = r'D:\Datasets\hpc_august\model_vectors.csv'
dataset = pd.read_csv(path,sep=',',header=0,low_memory=False,infer_datetime_format=True)

def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def baseline_MLP():
    nn=Sequential()
    nn.add(Dense(100,activation='relu',input_dim=3)) #3 cols
    nn.add(Dense(1))
    nn.compile(loss='mae',optimizer='adam')
    return nn

#print(dataset.head(15))

dataset['date']= pd.date_range(start='1/1/2015', periods=len(dataset), freq='M')
dataset = dataset.set_index('date')
#print(dataset.head(15))
#print(dataset.head(15))
dataset.plot()
pp.show()

upsampled = dataset.resample('D')
interpolated = upsampled.interpolate(method='linear')
#print(interpolated.head(15))
interpolated.plot()
pp.show()

print(dataset.shape)
print(interpolated.shape)

#CREATE TRAIN AND TEST SET

#XTEST,YTEST
X_test = dataset[['base','causal','similar']].values
Y_test = dataset['actual'].values
#print(X_test)
#print(Y_test)

#print(interpolated.head(10))

#XTRAIN,YTRAIN
for i in range(dataset.shape[0]):
    interpolated=interpolated.drop(dataset.index[i])

#print(interpolated.head(10))
#print(interpolated.shape)

X_train = interpolated[['base','causal','similar']].values
Y_train = interpolated['actual'].values
#print(X_train)
#print(Y_train)

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


model = LinearRegression().fit(X_train,Y_train)

Y_pred = model.predict(X_test)
#print('predicted response:', Y_pred, sep='\n')

print(mape(Y_test,Y_pred))

pp.plot(Y_pred, label='prediction')
pp.plot(Y_test, label='actual')
pp.legend()
pp.show()


nn=baseline_MLP()
es=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=170)
bestmodelpath = r'D:\Datasets\hpc_august\best_meta_model.h5'
mc = ModelCheckpoint(bestmodelpath, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
history=nn.fit(X_train,Y_train, epochs=4000, batch_size=72, validation_data=(X_test,Y_test), verbose=2, shuffle=False,callbacks=[es, mc])

saved_model = load_model(bestmodelpath)

pp.plot(history.history['loss'], label='train')
pp.plot(history.history['val_loss'], label='test')
pp.legend()
pp.savefig(r'D:\Datasets\hpc_august\combined_model_loss.jpg',dpi=300,bbox_inches="tight")
pp.show()

y_pred_mlp=saved_model.predict(X_test)

pp.plot(y_pred_mlp, label='prediction')
pp.plot(Y_test, label='actual')
pp.legend()
pp.show()

dataset['combined'] = y_pred_mlp
dataset.to_csv(r'D:\Datasets\hpc_august\model_combined.csv',index=False)
