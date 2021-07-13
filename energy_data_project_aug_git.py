# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:37:24 2021

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

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

path = r'D:\Datasets\hpc_august\db_power_consumption.csv'
price_path = r'D:\Datasets\hpc_august\energyprices_cedenar.xlsx'
dataset = pd.read_csv(path,sep=',',header=0,low_memory=False,infer_datetime_format=True)
price_dt=pd.read_excel(price_path)

#print(dataset.head(5))

#SORT BY MULTIPLE DEMOGRAPHIC CRITERIA IF NEEDED
#dataset=dataset.sort_values(by=['municipality','date','use'])




#print(price_dt.head())
#print(price_dt.dtypes)

#MAPPING CATEGORICAL VALUES TO NUMERICAL
cat_to_num={"area": {"U":0,"R":1}, 
            "municipality": {"BARBACOAS": 0, "CUMBITARA": 1, "EL ROSARIO": 2, "LEIVA": 3, "MAGUI": 4, "POLICARPA": 5, "ROBERTO PAYAN": 6},
            "use": {"Residential": 0, "Residential Sub": 1, "Industrial": 2, "Official": 3, "Commercial": 4, "Special": 5}}

#SEASONAL MAPPING: 0-WINTER(12-2), 1-SPRING(3-5),2-SUMMER(6-8),3-AUTUMN(9-11)
season_transform={"season": {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}}

dataset["price"]=0.0
dataset["tariff"]=0.0

#CREATE TOTAL PRICE FEATURE BASED ON USE, STRATUM AND DATE FROM CEDENAR CONVERTED TO EUROS
for j in range(dataset['code'].shape[0]):
    for i in range(price_dt['date'].shape[0]):
        sampledate=str(dataset['date'].values[j])
        pricedate=str(price_dt['date'].values[i])
        #print(sampledate,pricedate)
        if pricedate in sampledate:
            #print (True)
            if str(dataset['use'].values[j])=="Residential":
                dataset["price"].values[j]=price_dt['residential'].values[i]*0.00024*dataset['consumption'].values[j]
                dataset["tariff"].values[j]=price_dt['residential'].values[i]*0.00024
            if str(dataset['use'].values[j])=="Industrial":
                dataset["price"].values[j]=price_dt['commercial-industrial'].values[i]*0.00024*dataset['consumption'].values[j]
                dataset["tariff"].values[j]=price_dt['commercial-industrial'].values[i]*0.00024
            if str(dataset['use'].values[j])=="Commercial":
                dataset["price"].values[j]=price_dt['commercial-industrial'].values[i]*0.00024*dataset['consumption'].values[j]
                dataset["tariff"].values[j]=price_dt['commercial-industrial'].values[i]*0.00024
            if str(dataset['use'].values[j])=="Official":
                dataset["price"].values[j]=price_dt['official-special'].values[i]*0.00024*dataset['consumption'].values[j]
                dataset["tariff"].values[j]=price_dt['official-special'].values[i]*0.00024
            if str(dataset['use'].values[j])=="Special":
                dataset["price"].values[j]=price_dt['official-special'].values[i]*0.00024*dataset['consumption'].values[j]
                dataset["tariff"].values[j]=price_dt['official-special'].values[i]*0.00024
            if str(dataset['use'].values[j])=="Residential Sub":
                if dataset['stratum'].values[j]==0:
                    dataset["price"].values[j]=price_dt['residential-sub-stratum0'].values[i]*0.00024*dataset['consumption'].values[j]
                    dataset["tariff"].values[j]=price_dt['residential-sub-stratum0'].values[i]*0.00024
                if dataset['stratum'].values[j]==1:
                    dataset["price"].values[j]=price_dt['residential-sub-stratum1'].values[i]*0.00024*dataset['consumption'].values[j]
                    dataset["tariff"].values[j]=price_dt['residential-sub-stratum1'].values[i]*0.00024
                if dataset['stratum'].values[j]==2:
                    dataset["price"].values[j]=price_dt['residential-sub-stratum2'].values[i]*0.00024*dataset['consumption'].values[j]
                    dataset["tariff"].values[j]=price_dt['residential-sub-stratum2'].values[i]*0.00024
                if dataset['stratum'].values[j]==3:
                    dataset["price"].values[j]=price_dt['residential-sub-stratum3'].values[i]*0.00024*dataset['consumption'].values[j]
                    dataset["tariff"].values[j]=price_dt['residential-sub-stratum3'].values[i]*0.00024
        
        #break
    #break


#REPLACE CATEGORICAL WITH NUMERICAL
dataset = dataset.replace(cat_to_num)


#FIND THE REAL CLIENT CODE AND EXPLORE SEASONAL FEATURE
dataset["usertag"]=""
dataset["season"]=""
for m in range(dataset['code'].shape[0]):
    dataset["usertag"].values[m]=str(dataset['code'].values[m][0:2])+str(dataset['area'].values[m])+str(dataset['municipality'].values[m])+str(dataset['use'].values[m])+str(dataset['stratum'].values[m])
    dataset["season"].values[m]=datetime.datetime.strptime(dataset["date"].values[m],"%Y-%m-%d").month

dataset = dataset.replace(season_transform)

#IN CASE WE WANTED TO USE THE DEFAULT CODES
#dataset["code"] = dataset["code"].astype('category')
#dataset["usercode"] = dataset["code"].cat.codes

#HERE USERCODE IS DERIVED FROM THE FROM THE TAG THAT SHOWS THE REAL MAPPING TO USERS
dataset["usertag"] = dataset["usertag"].astype('category')
dataset["usercode"] = dataset["usertag"].cat.codes

#SHOW MEASUREMENTS SORTED FOR EACH USER
dataset=dataset.sort_values(by=['usercode','date'])


print(tabulate(dataset.head(10),headers='keys'))
print(dataset['usercode'].values[dataset['usercode'].shape[0]-1])
print(dataset.dtypes)

usrprice=[]
usrconsum=[]
usrdate=[]
tariffcurve=[]

#Lists containing all measurements of price, consumption and tariff for each user in each element
listusrprice=[]
listusrconsum=[]
listusrtariff=[]
listusrdate=[]


print(dataset['usercode'].values[dataset['usercode'].shape[0]-1],dataset['usertag'].shape[0])
print(dataset['usercode'].values[dataset['usercode'].shape[0]-1]+1)

for j in range(dataset['usercode'].values[dataset['usercode'].shape[0]-1]+1):
    for k in range(dataset['usertag'].shape[0]):
        if dataset['usercode'].values[k]==j:
            usrprice.append(dataset['price'].values[k])
            usrconsum.append(dataset['consumption'].values[k])
            usrdate.append(dataset['date'].values[k])
            tariffcurve.append(dataset['tariff'].values[k])
    listusrprice.append(usrprice)
    listusrconsum.append(usrconsum)
    listusrtariff.append(tariffcurve)
    listusrdate.append(usrdate)
    #print(usrprice)
    usrprice=[]
    usrconsum=[]
    usrdate=[]
    tariffcurve=[]
    #break

print(listusrdate[89],listusrconsum[89])

#dataset.to_csv(r'D:\Datasets\hpc_august\data_processed.csv',index=False)
#Plotting and printing user measurements (maybe align them with corresponding dates with a new list capturing the dates in the loop)
#for i in range(dataset['usercode'].values[dataset['usercode'].shape[0]-1]):
#    print(len(listusrprice[i]))
#fig = pp.figure(figsize=(50,2))        
#pp.plot(listusrdate[0],listusrprice[0])
#pp.show()
#pp.plot(listusrprice[1])
#pp.show()
#pp.plot(listusrprice[2])
#pp.show()

#FORM DATAFRAME FOR TIME SERIES CLUSTERING DEALING WITH VARYING COLUMNS
print(listusrdate[0],listusrconsum[0])

usrdf=pd.DataFrame()
usrdf['date']=price_dt['date']

#FORM PRICING PER CLIENT DATAFRAME
usrdftariff=pd.DataFrame()
usrdftariff['date']=price_dt['date']


print(listusrdate[0][0],listusrconsum[0][0])
print(len(listusrconsum),len(listusrconsum[0]))

#INITIALIZE FOR EACH USER
for n in range(dataset['usercode'].values[dataset['usercode'].shape[0]-1]+1):
    usrdf[str(n)]=np.NaN
    usrdftariff[str(n)]=0.0

#FILL IN THE VALUES
for n in range(dataset['usercode'].values[dataset['usercode'].shape[0]-1]+1):
    for k in range(len(listusrconsum[n])):
        for m in range(usrdf['date'].shape[0]):
            if usrdf['date'].values[m] in listusrdate[n][k]:
                usrdf[str(n)].values[m]=listusrconsum[n][k]
                usrdftariff[str(n)].values[m]=listusrtariff[n][k]

print(usrdf.head(35))
print(dataset['usercode'].values[dataset['usercode'].shape[0]-1],dataset['usertag'].shape[0],len(listusrconsum[n])-1)

print(usrdftariff.head(35))

usrdf.to_csv(r'D:\Datasets\hpc_august\consumption_perclient_series.csv',index=False)

usrdftariff.to_csv(r'D:\Datasets\hpc_august\tariff_perclient_series.csv',index=False)

#SCALE CONSUMPTION AND PRICE FEATURES TO THEIR RESPECTIVE ARRAYS
#X_train=np.array(dataset.drop(['code','date','usertag','usercode'],axis=1))
#print(X_train)

#scaler = preprocessing.StandardScaler().fit(X_train)
#X_scaledcons = scaler.transform(X_train)

