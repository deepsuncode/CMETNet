'''
 (c) Copyright 2022
 All rights reserved
 Programs written by Khalid A. Alobaid
 Department of Computer Science
 New Jersey Institute of Technology
 University Heights, Newark, NJ 07102, USA
 Permission to use, copy, modify, and distribute this
 software and its documentation for any purpose and without
 fee is hereby granted, provided that this copyright
 notice appears in all copies. Programmer(s) makes no
 representations about the suitability of this
 software for any purpose.  It is provided "as is" without
 express or implied warranty.
'''

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import datetime
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from statistics import median


def str_to_datatime(row):
    x = datetime.datetime.strptime(row,'%Y-%m-%d %H:%M:%S')
    return x

filename = './data/ICME_list.csv'
org_data = pd.read_csv(filename)
org_data["disturbance"] = [str_to_datatime(x) for x in org_data["disturbance"]]


year=2014
year_next = str(year+1)
year = str(year)
test_indices = org_data.index[(org_data['disturbance']>=datetime.datetime.fromisoformat(year+'-01-01 00:00:00'))&
                              (org_data['disturbance']<datetime.datetime.fromisoformat(year_next+'-01-01 00:00:00'))].tolist()


all_data = org_data[:]
all_data = all_data.drop(test_indices)
all_data.drop("disturbance", axis=1,inplace=True)
X_data = all_data[:]
X_data.drop("transit_time", axis=1,inplace=True)
y_data = all_data['transit_time']
data_shape=X_data.shape[1]
hold_out_data = org_data.iloc[test_indices, :]
events_disturbance = hold_out_data["disturbance"].reset_index(drop=True)
hold_out_data.drop("disturbance", axis=1,inplace=True)
hold_out_X_test = hold_out_data[:]
hold_out_X_test.drop("transit_time", axis=1,inplace=True)
hold_out_y_test = hold_out_data['transit_time']
hold_out_y_test_reseted = hold_out_y_test.reset_index(drop=True)


RF_model = RandomForestRegressor(n_estimators= 800, max_features= 'sqrt')
SVR_model = SVR(kernel ='rbf', cache_size =200)
XGBoost_model = XGBRegressor()
kernel = 1.0 * RBF() + WhiteKernel(noise_level=0.2) 
GPR_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30)


start_time = datetime.datetime.now()
training_times = 10

MAE_temp = 100
MAE_XGBoost_temp = 100
MAE_SVR_temp = 100
MAE_RF_temp = 100
MAE_GPR_temp = 100



report_each =1


for i in range(1,training_times+1):
    current_time = datetime.datetime.now()
    if i % report_each == 0:
        print("Epoch ",str(i).zfill(2),"/"+str(training_times))


    RF_X_train, RF_X_test, RF_y_train, RF_y_test = train_test_split(X_data, y_data, train_size=200)
    RF_regr = RF_model
    RF_result = RF_regr.fit(RF_X_train, RF_y_train).predict(hold_out_X_test)
    SVR_X_train, SVR_X_test, SVR_y_train, SVR_y_test = train_test_split(X_data, y_data, train_size=200)
    SVR_regr = SVR_model
    SVR_result = SVR_regr.fit(SVR_X_train, SVR_y_train).predict(hold_out_X_test)
    XGBoost_X_train, XGBoost_X_test, XGBoost_y_train, XGBoost_y_test = train_test_split(X_data, y_data, train_size=200)
    XGBoost_regr = XGBoost_model
    XGBoost_result = XGBoost_regr.fit(XGBoost_X_train, XGBoost_y_train).predict(hold_out_X_test)
    GPR_X_train, GPR_X_test, GPR_y_train, GPR_y_test = train_test_split(X_data, y_data, train_size=200)
    GPR_regr = GPR_model
    GPR_result = GPR_regr.fit(GPR_X_train, GPR_y_train).predict(hold_out_X_test)


    MAE_XGBoost_current = round(mean_absolute_error(hold_out_y_test, XGBoost_result),2)
    if MAE_XGBoost_current<MAE_XGBoost_temp:
        best_XGBoost=XGBoost_result
        best_XGBoost_regr = XGBoost_regr
        MAE_XGBoost_temp=MAE_XGBoost_current
        best_XGBoost_X_train = XGBoost_X_train

    MAE_SVR_current = round(mean_absolute_error(hold_out_y_test, SVR_result),2)
    if MAE_SVR_current<MAE_SVR_temp:
        best_SVR=SVR_result
        best_SVR_regr = SVR_regr
        MAE_SVR_temp=MAE_SVR_current
        best_SVR_X_train = SVR_X_train

    MAE_RF_current = round(mean_absolute_error(hold_out_y_test, RF_result),2)
    if MAE_RF_current<MAE_RF_temp:
        best_RF=RF_result
        best_RF_regr=RF_regr
        MAE_RF_temp=MAE_RF_current
        best_RF_X_train = RF_X_train 

    MAE_GPR_current = round(mean_absolute_error(hold_out_y_test, GPR_result),2)
    if MAE_GPR_current<MAE_GPR_temp:
        best_GPR=GPR_result
        best_GPR_regr=GPR_regr
        MAE_GPR_temp=MAE_GPR_current
        best_GPR_X_train = GPR_X_train


    median_AT = []
    ALL_TT = []

    

    for x in range(0,len(hold_out_y_test)):
        x1=best_XGBoost[x]
        x2=best_SVR[x]
        x3=best_RF[x]
        x4=best_GPR[x]
        All_results = [x1,x2,x3,x4]
        median_AT.append(median(All_results))
        ALL_TT.append(All_results)

        


    MAE_current = round(mean_absolute_error(hold_out_y_test, median_AT),2)

    if MAE_current<MAE_temp:
        MAE_temp=MAE_current  
        best_ALL_TT = ALL_TT

    del RF_regr
    del SVR_regr
    del XGBoost_regr
    del GPR_regr


path = './results/'   

#safe results to csv
np.savetxt(path+"Ensemble_model_"+year+"_COMB.csv", best_ALL_TT, delimiter=",")
np.savetxt(path+"Ensemble_model_"+year+"_y_test.csv", hold_out_y_test, delimiter=",")
events_disturbance.to_csv(path+"Ensemble_model_"+year+"_events_disturbance.csv", encoding='utf-8', index=False)

current_time = datetime.datetime.now()    
print("Done ------------------------------------ ")





