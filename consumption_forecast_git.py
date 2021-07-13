# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 12:01:19 2021

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
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter, FuncFormatter
import datetime
import statistics
import math
from scipy import stats

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

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
path = r'D:\Datasets\hpc_august\consumption_perclient_series.csv'
dataset = pd.read_csv(path,sep=',',header=0,low_memory=False,infer_datetime_format=True)
dataset=dataset[:-1]

#PER CLIENT TARIFF
path_tariff = r'D:\Datasets\hpc_august\tariff_perclient_series.csv'
dataset_tar = pd.read_csv(path_tariff,sep=',',header=0,low_memory=False,infer_datetime_format=True)
dataset_tar=dataset_tar[:-1]

#PROCESSED ENERGY DATA
path2 = r'D:\Datasets\hpc_august\data_processed.csv'
dataset2 = pd.read_csv(path2,sep=',',header=0,low_memory=False,infer_datetime_format=True)


#GRANGER CAUSALITY BETWEEN PAST TARIFF AND FUTURE CONSUMPTION
#COLUMN 0 = FUTURE CONSUMPTION, COLUMN 1 = PAST TARIFF 
#EXAMPLE ON CLIENT 1 AND GENERALISE ON ALL

#YEAR 2011 CONSUMPTION
#print(dataset['1'].values[1:13])

#YEAR 2011 TARIFF
#print(dataset_tar['1'].values[1:13])

causaltempdf=pd.DataFrame()
causaltempdf['1']=dataset['1'].values[1:13]
causaltempdf['2']=dataset_tar['1'].values[1:13]
#print(causaltempdf)
#causarray=causaltempdf[['1','2']].to_numpy()
#print(causarray)
maxlag=2
test = 'ssr_chi2test'

#PERFORM CAUSALITY TEST - GET MIN P VALUE: IF P<0.05 THEN THERE IS CAUSALITY
def test_causality(Xcauses,Yeffect):
        causaltempdf=pd.DataFrame()    
        causaltempdf['1']=Yeffect
        causaltempdf['2']=Xcauses
        causaltempdf=causaltempdf.fillna(0)
        #print(causaltempdf)
        try:
            gc_res=grangercausalitytests(causaltempdf, maxlag=maxlag, verbose=False)
            p_values = [round(gc_res[i+1][0][test][1],4) for i in range(maxlag)]
            result=np.min(p_values)
            return result
        except:
            return 5

years=5
for i in range(years):
    #print("FOR CONSUMPTION YEAR:",i)
    for j in range(i+1):
        causres=test_causality(dataset_tar['1'].values[1+j*12:13+j*12], dataset['1'].values[1+i*12:13+i*12])
        #print(causres)
#gc_res=grangercausalitytests(causaltempdf, maxlag=maxlag, verbose=False)
#p_values = [round(gc_res[i+1][0][test][1],4) for i in range(maxlag)]
#print(np.min(p_values))


#print(dataset.head(15))

dataset=dataset.drop(['date'],axis=1)
dataset=dataset.fillna(0.0)
#dataset=TimeSeriesScalerMeanVariance().fit_transform(dataset)

#LOOP THROUGH AND FIND SIMILARITY SCORES BETWEEN SEQUENCES
timeseries1=to_time_series(dataset["0"].values)
timeseries2=to_time_series(dataset["1"].values)
#print(timeseries1)

soft_score=soft_dtw(timeseries1, timeseries2, gamma=.1)
#print(soft_score)
#print(dataset.shape[0],dataset.shape[1])


#DTW MATRIX, LOWER VALUE=SERIES MORE SIMILAR
dtw_scoring_matrix=[]
for k in range(dataset.shape[1]):
    dtw_scores_col=[]
    timeseries1=to_time_series(dataset[str(k)].values)
    for m in range(dataset.shape[1]):
        timeseries2=to_time_series(dataset[str(m)].values)
        soft_score=soft_dtw(timeseries1, timeseries2, gamma=.1)
        dtw_scores_col.append(soft_score)
    dtw_scoring_matrix.append(dtw_scores_col)

#print(len(dtw_scoring_matrix),len(dtw_scoring_matrix[0]))

dtw_df=pd.DataFrame.from_records(dtw_scoring_matrix)

#print(dtw_df.head(10))

#FIND C MOST SIMILAR CLIENT TIMESERIES
c=4
similaritylist=[]
for k in range(dataset.shape[1]):
    arr=np.array(dtw_df[k].values)
    idx=np.argsort(arr)
    #print(idx[1:c])
    similaritylist.append(list(idx[1:c]))

#print(similaritylist)

#BUILD MULTICLASS CLASSIFIER with the introduction of similarity
seed = 7
np.random.seed(seed)
dataset2['class']=""

for i in range (dataset2.shape[0]):
    dataset2['class'].values[i]=str(dataset2['use'].values[i])+str(dataset2['stratum'].values[i])


#ADD IMPACT OF SIMILARITY LIST
dataset2['similar0']=0.0
dataset2['similar1']=0.0
dataset2['similar2']=0.0


dataset=dataset.dropna()

for i in range(dataset2.shape[0]):
    code=dataset2['usercode'].values[i]
    for k in range(len(similaritylist[code])):
        
        #print(similaritylist[code][k])
        valuelist=dataset[str(similaritylist[code][k])].values
        valuelist=[m for m in valuelist if m!=0]
        meancons=statistics.mean(valuelist)
        #print(valuelist,meancons)
        dataset2['similar'+str(k)].values[i]=meancons
    #break
    

print("PRICE FOR CONSUMER")
print(dataset_tar['1'].values[1:13+4*12])


"""
#BASE CLASSIFIER UP TO TEST LOSS and ACC WORKING WITH LOW ACC

X=dataset2.drop(['Unnamed: 0','code','date','usertag','usercode','price','tariff','class','use','stratum'],axis=1)
Y=dataset2['class']


print(tabulate(X.head(10),headers='keys'))
print(tabulate(X.tail(10),headers='keys'))

print(Y.head(10))


encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
onehot_Y = tf.keras.utils.to_categorical(encoded_Y)
print(onehot_Y)
print(onehot_Y.shape)

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, onehot_Y, test_size=0.3, random_state=2)

n_features = X.shape[1]
n_classes = onehot_Y.shape[1]

def client_classifier_model():
    model = Sequential()
    model.add(Dense(X.shape[1], input_dim = X.shape[1], activation='relu'))
    model.add(Dense(onehot_Y.shape[1],activation='sigmoid'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

model=client_classifier_model()
model.fit(X_train, Y_train,batch_size=32,epochs=1000,verbose=0,validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
"""


#client_classifier = KerasClassifier(build_fn=client_classifier_model, nb_epoch=20,batch_size=5, verbose=0)
#kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(client_classifier, X, onehot_Y, cv=kfold)
#print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


#CREATE BASE LSTM FOR USER CONSUMPTION FORECASTING

def calculate_mape(Y_actual, Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

#DEFINE MAPE CALCULATION
def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

#SEQUENCE PREPARATION
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

#MULTIVARIATE FORMULATION
def multivariate_supervised_split(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)
#APPROACH-1
#AUGMENT FEATURE SET WITH ADDITIONAL INPUT FROM SIMILARITY LIST
#UNSUCCESSFUL: MODEL COULD NOT LEARN FROM NEIGHBORS IN THAT SETUP


#ADFULLER STATIONARITY CHECK
def stationarity_function(dataset,Xlstm,Ylstm):
    #CHECK STATIONARITY OF FEATURES - ADFULLER
    #stationarity of consumption
    stationarity_cons=adfuller(dataset['1'].values[1:13+4*12])
    print("Consumption has ADF: %f p-value: %f" % (stationarity_cons[0], stationarity_cons[1]))
    
    #TEST STATIONARITY OF INPUTS (only t-1 is non stationary)
    for i in range(Xlstm.shape[1]):
        stationarity=adfuller(Xlstm[:,i])
        print('ADF statistic: %f p-value: %f' % (stationarity[0],stationarity[1]))
    
    #TEST STATIONARITY OF OUTPUTS
    #Ylstm=Ylstm.reshape(-1,1)        
    #stationarity=adfuller(Ylstm)
    #print('ADF statistic: %f p-value: %f' % (stationarity[0],stationarity[1]))


#stXlstm,nYlstm will be used for input and output of the LSTM model below instead of Xlstm and Ylstm
#SPLIT stXlstm and nYlstm into train and test 75-25 THEN SCALES TRAINING DATA

def dtw_similarity_function(dataset, similaritylist, Xlstm, Ylstm, Xlstm_train, Xlstm_test, Ylstm_train, Ylstm_test, n_steps, user, flag):
    print(similaritylist[user])
    
    for i in range(len(similaritylist[user])):
        similarX, similarY = split_sequence(dataset[str(similaritylist[user][i])].values[1:13+4*12], n_steps)
        similarX_train, similarX_test, similarY_train, similarY_test = train_test_split(similarX, similarY, test_size=0.25, shuffle=False)
        if flag == 0:
            Xlstm_train = np.concatenate((Xlstm_train, similarX_train), axis = 1)
            Xlstm_test = np.concatenate((Xlstm_test, similarX_test), axis = 1)
            return Xlstm_train, Xlstm_test
        else:
            Xlstm_train = np.concatenate((Xlstm_train, similarX_train), axis = 0)
            Ylstm_train = np.concatenate((Ylstm_train, similarY_train), axis = 0)
            return Xlstm_train, Ylstm_train
    
    #print(Xlstm_train)
        
    

def causality_function(dataset, similaritylist, Xlstm, Ylstm, Xlstm_train, Xlstm_test, Ylstm_train, Ylstm_test, n_steps, flag):
    #APPROACH 2: FIND BEST "SET" FROM NEIGHBORS TO ATTACH TO CONSUMER 1
    
    print(similaritylist[1])
    causalitylist=[]
    for i in range(89):
        #print("This is similarity formulation of neighbor: ",i)
        similarX, similarY = split_sequence(dataset[str(i)].values[1:13+4*12], n_steps)
        similarX_train, similarX_test, similarY_train, similarY_test = train_test_split(similarX, similarY, test_size=0.25, shuffle=False)
        #print(similarX_train.shape,Ylstm_train.shape)
        for k in range(similarX_train.shape[1]):
            cresult = test_causality(similarX_train[:,k],Ylstm_train)
            if cresult<0.05:
                #print(cresult)
                #print("Neighbor:", i)
                causalitylist.append([cresult,i,k])
        #for j in range(len(similarX)):
        #    print(similarX[j], similarY[j])
        #similarY=similarY.reshape(-1,1) 
        #Xlstm = np.concatenate((Xlstm,similarX,similarY),axis=1)
    print("CAUSALITY LIST")    
    print(causalitylist)
    #print("CAUSAL NEIGHBORS")
    print("TRAINING SET BEFORE:")
    print(Xlstm_train[0])
    for m in range(len(causalitylist)):
        #print(causalitylist[m][1],causalitylist[m][2])
        neighborX, neighborY = split_sequence(dataset[str(causalitylist[m][1])].values[1:13+4*12], n_steps)
        neighborX_train, neighborX_test, neighborY_train, neighborY_test = train_test_split(similarX, similarY, test_size=0.25, shuffle=False)
        if flag == 0:
            extra_column_train = np.array(neighborX_train[:,causalitylist[m][2]], copy=False, subok=True, ndmin=2).T
            extra_column_test = np.array(neighborX_test[:,causalitylist[m][2]], copy=False, subok=True, ndmin=2).T
        #print(extra_column.shape)
            Xlstm_train = np.column_stack([Xlstm_train,extra_column_train])
            Xlstm_test = np.column_stack([Xlstm_test,extra_column_test])
        else:
            Xlstm_train = np.concatenate((Xlstm_train, neighborX_train), axis = 0)
            Ylstm_train = np.concatenate((Ylstm_train, neighborY_train), axis = 0)
    print("TRAINING SET AFTER")
    print(Xlstm_train[0])
    print(Xlstm_train.shape)
    if flag == 0:
        return Xlstm_train,Xlstm_test
    else:
        return Xlstm_train,Ylstm_train
    

#print (Xlstm_train, Xlstm_test, Xlstm_train.shape)
#SCALE FEATURES BASED ON SHAPIRO TEST
def feature_scaling_lstm(Xlstm_train, Xlstm_test, Ylstm_train, Ylstm_test):
    #TEST IF INPUTS HAVE A NORMAL DISTRIBUTION. IF NORMAL->STANDARDIZE, ELSE NORMALIZE
    #TEST the training INPUTS
    #IT IS BETTER TO SCALE EACH FEATURE INDIVIDUALLY
    stscaler=StandardScaler()
    normalscaler=MinMaxScaler()
    stXlstm_train=Xlstm_train
    stXlstm_test=Xlstm_test
    #print(stXlstm_train)
    #print(stXlstm_test)
    for i in range(Xlstm_train.shape[1]):
        shapiro_test = stats.shapiro(Xlstm_train[:,i])
        print(shapiro_test.pvalue)
        feature_train=Xlstm_train[:,i].reshape(-1,1)
        feature_test=Xlstm_test[:,i].reshape(-1,1)
        if shapiro_test.pvalue > 0.05:
            print("Feature standardization")
            stscaler.fit(feature_train)
            scaled_train = stscaler.transform(feature_train)
            stscaler.fit(feature_test)
            scaled_test = stscaler.transform(feature_test)
        else:
            print("Feature normalization")
            normalscaler.fit(feature_train)
            scaled_train = normalscaler.transform(feature_train)
            normalscaler.fit(feature_test)
            scaled_test = normalscaler.transform(feature_test)
        #print(scaled_train)
        #print(scaled_test)
        #print(stXlstm_test.shape[0])
        #print(stXlstm_test)
        #print(stXlstm_train.shape[0])
        for k in range (stXlstm_train.shape[0]):
            stXlstm_train[k][i]=scaled_train[k][0]
            
        for j in range (stXlstm_test.shape[0]):
            stXlstm_test[j][i]=scaled_test[j][0]
        #print(scaled_test[1][0])
        #print(stXlstm_test)
        #break
    #print(stXlstm_train)
    #print(stXlstm_test)
    
    #INPUTS ARE NORMAL->NEED STANDARDIZATION
    #HYPOTHETICAL FOR ALL FEATURES (NOT NEEDED NOW)
    #stscaler = stscaler.fit( Xlstm_train)
    #stXlstm_train=stscaler.transform(Xlstm_train)
    #print(stXlstm_train)
    #SAME SCALING MUST APPLY TO TEST SET
    #stXlstm_test = stscaler.transform(Xlstm_test)
    
    #TEST the training OUTPUTS
    shapiro_test = stats.shapiro(Ylstm_train)
    print(shapiro_test.pvalue)
    #OUTPUT training set NORMAL -> STANDARDIZE AS WELL
    Ylstm_train=Ylstm_train.reshape(-1,1)  
    nscaler=StandardScaler()
    nscaler=nscaler.fit(Ylstm_train)
    nYlstm_train = nscaler.transform(Ylstm_train)
    #print(nYlstm_train)
    #SAME SCALING MUST APPLY TO TEST OUTPUT SET
    Ylstm_test=Ylstm_test.reshape(-1,1)
    nscaler=nscaler.fit(Ylstm_test)  
    nYlstm_test = nscaler.transform(Ylstm_test)
    
    """
    #FOR FINDINGS GATHERING PUT IN COMMENTS OTHERWISE###
    path_model = r'D:\Datasets\hpc_august\model_combined.csv'
    dataset_model = pd.read_csv(path_model,sep=',',header=0,low_memory=False,infer_datetime_format=True)
    
    path_model_bc = r'D:\Datasets\hpc_august\model_base_causal.csv'
    dataset_model_bc = pd.read_csv(path_model_bc,sep=',',header=0,low_memory=False,infer_datetime_format=True)
    
    path_model_bs = r'D:\Datasets\hpc_august\model_base_similar.csv'
    dataset_model_bs = pd.read_csv(path_model_bs,sep=',',header=0,low_memory=False,infer_datetime_format=True)
    
    path_model_cs = r'D:\Datasets\hpc_august\model_causal_similar.csv'
    dataset_model_cs = pd.read_csv(path_model_cs,sep=',',header=0,low_memory=False,infer_datetime_format=True)
    
    
    
    base_inverted = nscaler.inverse_transform(dataset_model['base'])
    similar_inverted = nscaler.inverse_transform(dataset_model['similar'])
    causal_inverted = nscaler.inverse_transform(dataset_model['causal'])
    actual_inverted = nscaler.inverse_transform(dataset_model['actual'])
    combined_inverted = nscaler.inverse_transform(dataset_model['combined'])
    
    bc_inverted = nscaler.inverse_transform(dataset_model_bc['base_causal'])
    bs_inverted = nscaler.inverse_transform(dataset_model_bs['base_similar'])
    cs_inverted = nscaler.inverse_transform(dataset_model_cs['causal_similar'])
    print(base_inverted)
    
    
    #PLOT 2C MODELS
    fig,ax = pp.subplots()
    pp.plot(bc_inverted, label='base_causal')
    pp.plot(bs_inverted, label='base_similar')
    pp.plot(cs_inverted, label='causal_similar')
    pp.plot(combined_inverted, label = 'base_causal_similar')
    pp.plot(actual_inverted, label='actual_consumption')
    ax.get_yaxis().set_major_formatter(pp.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    pp.legend()
    pp.xlabel('Data Point')
    pp.ylabel('Power Consumption (kWh)')
    pp.savefig(r'D:\Datasets\hpc_august\pairwise_prediction.jpg',dpi=300,bbox_inches="tight")
    pp.show()
    
    
    
    #METRICS FOR STANDALONE COMPONENTS AND COMBINATORIAL MODEL
    #MAPE
    mape_base = calculate_mape(actual_inverted, base_inverted)
    mape_causal = calculate_mape(actual_inverted, causal_inverted)
    mape_similar = calculate_mape(actual_inverted, similar_inverted)
    mape_combined = calculate_mape(actual_inverted, combined_inverted)
    print("MAPE BASE: ", mape_base)
    print("MAPE CAUSAL: ", mape_causal)
    print("MAPE SIMILAR: ", mape_similar)
    print("MAPE COMBINED: ", mape_combined)
    
    #MSE
    mse_base = mean_squared_error(actual_inverted, base_inverted)
    mse_causal = mean_squared_error(actual_inverted, causal_inverted)
    mse_similar = mean_squared_error(actual_inverted, similar_inverted)
    mse_combined = mean_squared_error(actual_inverted, combined_inverted)
    print("MSE BASE: ", mse_base)
    print("MSE CAUSAL: ", mse_causal)
    print("MSE SIMILAR: ", mse_similar)
    print("MSE COMBINED: ", mse_combined)
    
    #RMSE
    rmse_base = math.sqrt(mse_base)
    rmse_causal = math.sqrt(mse_causal)
    rmse_similar = math.sqrt(mse_similar)
    rmse_combined = math.sqrt(mse_combined)
    print("RMSE BASE: ", rmse_base)
    print("RMSE CAUSAL: ", rmse_causal)
    print("RMSE SIMILAR: ", rmse_similar)
    print("RMSE COMBINED: ", rmse_combined)
    
    #MAE
    mae_base = mean_absolute_error(actual_inverted, base_inverted)
    mae_causal = mean_absolute_error(actual_inverted, causal_inverted)
    mae_similar = mean_absolute_error(actual_inverted, similar_inverted)
    mae_combined = mean_absolute_error(actual_inverted, combined_inverted)
    print("MAE BASE: ", mae_base)
    print("MAE CAUSAL: ", mae_causal)
    print("MAE SIMILAR: ", mae_similar)
    print("MAE COMBINED: ", mae_combined)
    
    
    #METRICS FOR 2-COMPONENT VARIATIONS
    #MAPE 2-C
    mape_bc= calculate_mape(actual_inverted, bc_inverted)
    mape_bs = calculate_mape(actual_inverted, bs_inverted)
    mape_cs = calculate_mape(actual_inverted, cs_inverted)
    print("MAPE BC: ", mape_bc)
    print("MAPE BS: ", mape_bs)
    print("MAPE CS: ", mape_cs)
    
    #MSE 2-C
    mse_bc= mean_squared_error(actual_inverted, bc_inverted)
    mse_bs = mean_squared_error(actual_inverted, bs_inverted)
    mse_cs = mean_squared_error(actual_inverted, cs_inverted)
    print("MSE BC: ", mse_bc)
    print("MSE BS: ", mse_bs)
    print("MSE CS: ", mse_cs)
    
    #RMSE 2-C
    rmse_bc= math.sqrt(mse_bc)
    rmse_bs = math.sqrt(mse_bs)
    rmse_cs = math.sqrt(mse_cs)
    print("RMSE BC: ", rmse_bc)
    print("RMSE BS: ", rmse_bs)
    print("RMSE CS: ", rmse_cs)
    
    #MSE 2-C
    mae_bc= mean_absolute_error(actual_inverted, bc_inverted)
    mae_bs = mean_absolute_error(actual_inverted, bs_inverted)
    mae_cs = mean_absolute_error(actual_inverted, cs_inverted)
    print("MAE BC: ", mae_bc)
    print("MAE BS: ", mae_bs)
    print("MAE CS: ", mae_cs)
    
    
    pp.plot(base_inverted, label='base_component')
    pp.plot(similar_inverted, label='similar_component')
    pp.plot(causal_inverted, label='causal_component')
    pp.plot(actual_inverted, label='actual_consumption')
    pp.legend()
    pp.xlabel('Data Point')
    pp.ylabel('Kilowatt-hour (kWh)')
    pp.savefig(r'D:\Datasets\hpc_august\base_prediction.jpg',dpi=300,bbox_inches="tight")
    pp.show()
    
    
    pp.plot(combined_inverted, label='MLP_combinatorial')
    pp.plot(actual_inverted, label='actual_consumption')
    pp.legend()
    pp.xlabel('Data Point')
    pp.ylabel('Kilowatt-hour (kWh)')
    pp.savefig(r'D:\Datasets\hpc_august\mlp_actual_prediction.jpg',dpi=300,bbox_inches="tight")
    pp.show()
    
    
    fig,ax = pp.subplots()
    pp.plot(base_inverted, label='base_component')
    pp.plot(similar_inverted, label='similar_component')
    pp.plot(causal_inverted, label='causal_component')
    pp.plot(combined_inverted, label='MLP_combinatorial')
    pp.plot(actual_inverted, label='actual_consumption')
    pp.legend()
    ax.get_yaxis().set_major_formatter(pp.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    pp.xlabel('Data Point')
    pp.ylabel('Power Consumption (kWh)')
    pp.savefig(r'D:\Datasets\hpc_august\all_predictions_together_formatted.jpg',dpi=300,bbox_inches="tight")
    pp.show()
    """
    
    
    return stXlstm_train, stXlstm_test, nYlstm_train, nYlstm_test

#SINCE TRAINING VALUES NEEDED TO BE STANDARDIZED ->STANDARDIZE TEST SET VALUES TO FEED THE MODEL



#BASE LSTM MODEL DEFINITION STARTS HERE
#Lstm with only features for 1 user has input shape=(n_steps, n_features)
#Approach 1: n_steps changes to the dimension of input
#Predicts next month based on n_steps previous months, Integrates Early Stopping
def model_structure(stXlstm_test,stXlstm_train,nYlstm_train,nYlstm_test,Ylstm_train, Ylstm_test,units,n_steps,n_features):
    #model = Sequential()
    
    #for stationary LSTM layer
    #model.add(LSTM(4, batch_input_shape=(1, stXlstm_train.shape[1], stXlstm_train.shape[2]), stateful=True))
    
    #for non stationary
    #model.add(LSTM(units, input_shape=(stXlstm_train.shape[1], stXlstm_train.shape[2])))
    #model.add(Dense(1))
    #model.compile(optimizer='adam', loss='mse')
    
    model= Sequential()
    #First Layer
    model.add(LSTM(100,input_shape=(stXlstm_train.shape[1], stXlstm_train.shape[2])))
    #Output Layer
    model.add(Dense(1))
    #Compile, choose loss and optimizer metrics
    model.compile(loss='mae',optimizer='adam')
    
    #Implement Early Stopping for non stationary
    es=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=170)
    bestmodelpath = r'D:\Datasets\hpc_august\best_lstm.h5'
    mc = ModelCheckpoint(bestmodelpath, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    #Fit Model on training data for non stationary
    history=model.fit(stXlstm_train,nYlstm_train, batch_size=72,validation_data=(stXlstm_test,nYlstm_test),epochs=4000,verbose=2,callbacks=[es, mc])
    
    #Evaluate model with built-in function (non-stationary)
    lstmresults_train=model.evaluate(stXlstm_train,nYlstm_train,verbose=0)
    lstmresults_test=model.evaluate(stXlstm_test,nYlstm_test,verbose=0)
    print(lstmresults_train)
    print(lstmresults_test)
    
    #Evaluate best model with built-in functions (non-stationary)
    saved_model = load_model(bestmodelpath)
    bestlstmresults_train = saved_model.evaluate(stXlstm_train,nYlstm_train, verbose=0)
    bestlstmresults_test = saved_model.evaluate(stXlstm_test,nYlstm_test, verbose=0)
    print(bestlstmresults_train)
    print(bestlstmresults_test)
    
    
    # plot training history
    pp.plot(history.history['loss'], label='train')
    pp.plot(history.history['val_loss'], label='test')
    pp.legend()
    pp.show()
    
    #Make and review predictions
    #(for the stationary version, predict function needs batch_size=1)
    #Prediction on Train Set
    trainPredict = saved_model.predict(stXlstm_train)
    #trainPredict = model.predict(stXlstm_train)
    #Prediction on Test Set
    testPredict = saved_model.predict(stXlstm_test)
    #testPredict = model.predict(stXlstm_test)
    
    #print(nYlstm_train,trainPredict)
    
    #Plot Predictions on training set
    #pp.plot(nYlstm_train)
    #pp.plot(trainPredict)
    #pp.show()
    
    #Invert the transformation OF TRAINING SET
    Ylstm_train=Ylstm_train.reshape(-1,1)
    nscaler=StandardScaler()
    nscaler=nscaler.fit(Ylstm_train)
    trainPredict_inv = nscaler.inverse_transform(trainPredict)
    nYlstm_train_inv = nscaler.inverse_transform(nYlstm_train)
    
    # calculate root mean squared error OF TRAINING
    trainScore = math.sqrt(mean_squared_error(trainPredict_inv, nYlstm_train_inv))
    print('Train Score: %.2f RMSE' % (trainScore))
    
    #CALCULATE TRAINING MAPE
    trainingmape=mape(nYlstm_train_inv, trainPredict_inv)
    print("TRAINING MAPE")
    print(trainingmape)
    
    #pp.plot(nYlstm_train_inv)
    #pp.plot(trainPredict_inv)
    #pp.show()
    
    #Plot Predictions on TEST set
    #pp.plot(nYlstm_test)
    #pp.plot(testPredict)
    #pp.show()
    
    #Invert the transformation OF TEST SET
    Ylstm_test=Ylstm_test.reshape(-1,1)
    nscaler=nscaler.fit(Ylstm_test)
    testPredict_inv = nscaler.inverse_transform(testPredict)
    nYlstm_test_inv = nscaler.inverse_transform(nYlstm_test)
    
    # calculate root mean squared error OF TEST
    testScore = math.sqrt(mean_squared_error(testPredict_inv, nYlstm_test_inv))
    print('Test Score: %.2f RMSE' % (testScore))
    
    #CALCULATE TEST MAPE
    testmape=mape(nYlstm_test_inv, testPredict_inv)
    print(testmape)
    
    #pp.plot(nYlstm_test_inv)
    #pp.plot(testPredict_inv)
    #pp.show()
    return saved_model

#Arguements: ensemble models, weights for average, test set
#GET FORECAST FROM ENSEMBLE (WITH OR WITHOUT WEIGHTS)
def ensemble_forecast(members,weights,stXlstm_test):
    forecasts = [model.predict(stXlstm_test) for model in members]
    forecasts = np.array(forecasts)
    #print(forecasts)
    #Median or Weighted Average
    #processed_forecasts = np.median(forecasts,axis=0)
    if weights == None:
        processed_forecasts = np.median(forecasts,axis=0)
    else:
        processed_forecasts = np.average(forecasts,axis=0,weights=weights)
    
    return processed_forecasts

#GET ENSEMBLE MAPE ( POTENTIAL CHOICE OF A SUBSET OF MEMBERS)
def evaluate_ensemble(members, howmany,weights, stXlstm_test, nYlstm_test):
    subset = members[:howmany]
    forecasts = ensemble_forecast(subset,weights,stXlstm_test)
    evaluation = mape(nYlstm_test, forecasts)
    return evaluation

#GRID SEARCH FOR ENSEMBLE WEIGHTS
def grid_search(members, stXlstm_test, nYlstm_test):
    w=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    best_evaluation, best_weights = np.inf, None
    for weights in product(w, repeat=len(members)):
        if len(set(weights)) ==1:
            continue
        
        evaluation = evaluate_ensemble(members, 2, weights, stXlstm_test, nYlstm_test)
        if evaluation < best_evaluation:
            best_evaluation, best_weights = evaluation, weights
    return list(best_weights)


def aggregate_model(Xlstm_train, Xlstm_test, Ylstm_train, Ylstm_test,n_steps,n_features):
    #GET THE SCALED/RESHAPED TRAIN AND TEST SETS FOR LSTM MODEL ENSEMBLE
    stXlstm_train, stXlstm_test, nYlstm_train, nYlstm_test = feature_scaling_lstm(Xlstm_train, Xlstm_test, Ylstm_train, Ylstm_test)
    #Reshape input from [samples, timesteps] to [samples, timesteps, features] - READY FOR LSTM - multi-timestep formulation
    #stXlstm_train = stXlstm_train.reshape((stXlstm_train.shape[0], stXlstm_train.shape[1], n_features)) 
    #stXlstm_test = stXlstm_test.reshape((stXlstm_test.shape[0], stXlstm_test.shape[1], n_features))
    
    #Reshape input from [samples, timesteps] to [samples, timesteps, features] - READY FOR LSTM - single timestep formulation works better
    
    stXlstm_train = stXlstm_train.reshape((stXlstm_train.shape[0],1,stXlstm_train.shape[1]))
    stXlstm_test = stXlstm_test.reshape((stXlstm_test.shape[0],1,stXlstm_test.shape[1]))   
    
    #print(stXlstm_train.shape)
    #print(stXlstm_test.shape)
    #print(nYlstm_train.shape)
    #print(nYlstm_test.shape)
    
    #print(stXlstm_test)
    #print(nYlstm_test)
    
    #ENSEMBLE WITH CAUSALITY (No noticeable improvement on this dataset)
    #members = [model_structure(stXlstm_test,stXlstm_train,nYlstm_train,nYlstm_test,Ylstm_train,Ylstm_test,4,stXlstm_train.shape[1],n_features) for _ in range(n_members)]
    
    #ENSEMBLE WITH SIMILARITY (Negative Impact on training with those extra features)
    #members = [model_structure(stXlstm_test,stXlstm_train,nYlstm_train,nYlstm_test,Ylstm_train,Ylstm_test,4,stXlstm_train.shape[1],n_features) for _ in range(n_members)]
    
    
    #CREATE ENSEMBLE OF LSTMs FOR THE FRAMED PROBLEM - NORMAL EXECUTION
    members = [model_structure(stXlstm_test,stXlstm_train,nYlstm_train,nYlstm_test,Ylstm_train,Ylstm_test,100,n_steps,n_features) for _ in range(n_members)]
    
    weights = grid_search(members, stXlstm_test, nYlstm_test)
    ensemble_results = ensemble_forecast(members,weights,stXlstm_test)
    print(ensemble_results)
    print(weights)
    return ensemble_results, weights, stXlstm_train, stXlstm_test, nYlstm_train, nYlstm_test, members
    




###  MAIN PROJECT BODY ###

#DEFINE SUPERVISED LEARNING PROBLEM PARAMETERS
n_steps=3
n_features=1
n_members=2
flag=0


#DEFAULT DATA FOR ONE CLIENT CHOSEN TO MODEL INTO INPUT/OUTPUT FEATURES
Xlstm, Ylstm = split_sequence(dataset['1'].values[1:13+4*12], n_steps)
print("Input / Output features")
for i in range(len(Xlstm)):
	print(Xlstm[i], Ylstm[i])
print(Xlstm.shape)


#TRAIN - TEST SPLIT
Xlstm_train, Xlstm_test, Ylstm_train, Ylstm_test = train_test_split(Xlstm, Ylstm, test_size=0.25, shuffle=False)
print(Xlstm_train.shape, Xlstm_test.shape,Ylstm_train.shape, Ylstm_test.shape)




#NORMAL EXECUTION
ensemble_results, weights, stXlstm_train, stXlstm_test, nYlstm_train, nYlstm_test, members = aggregate_model(Xlstm_train, Xlstm_test, Ylstm_train, Ylstm_test,n_steps,n_features)

#FOR RESEARCH FINDINGS WITHOUT RETURN
#aggregate_model(Xlstm_train, Xlstm_test, Ylstm_train, Ylstm_test,n_steps,n_features)



#PLOT ENSEMBLE RESULTS WITH GRID SEARCH WEIGHTS - GET BASE ENSEMBLE RESULTS VECTOR
pp.plot(nYlstm_test,label='test-base')
pp.plot(ensemble_results, label='predict-base')
pp.legend()
pp.show()

#PLOT ENSEMBLE RESULTS WITHOUT WEIGHTS
#ensemble_results = ensemble_forecast(members,None,stXlstm_test)
#pp.plot(nYlstm_test, label='test-base')
#pp.plot(ensemble_results, label='predict-base')
#pp.legend()
#pp.show()

#CAUSALITY EXECUTION - GET CAUSALITY VECTOR
#EXPLORE CAUSALITY
if flag == 0:
    Xlstm_train_causal, Xlstm_test_causal=causality_function(dataset, similaritylist, Xlstm, Ylstm, Xlstm_train, Xlstm_test, Ylstm_train, Ylstm_test, n_steps, flag)
    ensemble_results_causal, weights_causal, stXlstm_train_causal, stXlstm_test_causal, nYlstm_train_causal, nYlstm_test_causal, members_causal = aggregate_model(Xlstm_train_causal, Xlstm_test_causal, Ylstm_train, Ylstm_test,n_steps,n_features)

else:
    #CAUSALITY ROW-WISE = NEGATIVE IMPACT
    Xlstm_train_causal, Ylstm_train_causal=causality_function(dataset, similaritylist, Xlstm, Ylstm, Xlstm_train, Xlstm_test, Ylstm_train, Ylstm_test, n_steps, flag)
    ensemble_results_causal, weights_causal, stXlstm_train_causal, stXlstm_test_causal, nYlstm_train_causal, nYlstm_test_causal, members_causal = aggregate_model(Xlstm_train_causal, Xlstm_test, Ylstm_train_causal, Ylstm_test,n_steps,n_features)

#ensemble_results_causal, weights_causal, stXlstm_train_causal, stXlstm_test_causal, nYlstm_train_causal, nYlstm_test_causal, members_causal = aggregate_model(Xlstm_train, Xlstm_test, Ylstm_train, Ylstm_test,n_steps,n_features)


#PLOT ENSEMBLE RESULTS WITH GRID SEARCH WEIGHTS
pp.plot(nYlstm_test,label='test-causal')
pp.plot(ensemble_results_causal, label='predict-causal')
pp.legend()
pp.show()


#PLOT ENSEMBLE RESULTS WITHOUT WEIGHTS
#ensemble_results_causal = ensemble_forecast(members_causal,None,stXlstm_test)
#pp.plot(nYlstm_test)
#pp.plot(ensemble_results_causal)
#pp.show()

#SIMILARITY EXECUTION - GET SIMILARITY VECTOR
#EXPLORE SIMILARITY
if flag == 0:
    Xlstm_train_similar, Xlstm_test_similar =dtw_similarity_function(dataset, similaritylist, Xlstm, Ylstm, Xlstm_train, Xlstm_test, Ylstm_train, Ylstm_test, n_steps, 1, flag)
    ensemble_results_similar, weights_similar, stXlstm_train_similar, stXlstm_test_similar, nYlstm_train_similar, nYlstm_test_similar, members_similar = aggregate_model(Xlstm_train_similar, Xlstm_test_similar, Ylstm_train, Ylstm_test,n_steps,n_features)

else:
     Xlstm_train_similar, Ylstm_train_similar =dtw_similarity_function(dataset, similaritylist, Xlstm, Ylstm, Xlstm_train, Xlstm_test, Ylstm_train, Ylstm_test, n_steps, 1, flag)
     ensemble_results_similar, weights_similar, stXlstm_train_similar, stXlstm_test_similar, nYlstm_train_similar, nYlstm_test_similar, members_similar = aggregate_model(Xlstm_train_similar, Xlstm_test, Ylstm_train_similar, Ylstm_test,n_steps,n_features)

pp.plot(nYlstm_test,label='test-similar')
pp.plot(ensemble_results_similar, label='predict-similar')
pp.legend()
pp.show()

#REGRESSION MODEL THAT FITS BASE, CAUSAL AND SIMILAR ELEMENT TO PREDICTION
print(ensemble_results)
print(ensemble_results_causal)
print(ensemble_results_similar)
print(nYlstm_test)


model_vectors=pd.DataFrame()
model_vectors['base']=pd.Series(ensemble_results.flatten())
model_vectors['causal']=pd.Series(ensemble_results_causal.flatten())
model_vectors['similar']=pd.Series(ensemble_results_similar.flatten())
model_vectors['actual']=pd.Series(nYlstm_test.flatten())



#model_vectors.to_csv(r'D:\Datasets\hpc_august\model_vectors.csv',index=False)

#MAIN PROJECT BODY ENDS HERE


