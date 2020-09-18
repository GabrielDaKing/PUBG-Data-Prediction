# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 15:38:47 2019

@author: gncis
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


housing_data = pd.read_csv("housing.csv")

label_encoder = LabelEncoder()
housing_data.iloc[:,5] =label_encoder.fit_transform(housing_data.iloc[:,5])
housing_data.iloc[:,6] =label_encoder.fit_transform(housing_data.iloc[:,6])
housing_data.iloc[:,7] =label_encoder.fit_transform(housing_data.iloc[:,7])
housing_data.iloc[:,8] =label_encoder.fit_transform(housing_data.iloc[:,8])
housing_data.iloc[:,9] =label_encoder.fit_transform(housing_data.iloc[:,9])
housing_data.iloc[:,11] =label_encoder.fit_transform(housing_data.iloc[:,11]) 

dummy =pd.get_dummies(housing_data['furnishingstatus'])
housing_data = pd.concat([housing_data,dummy],axis=1)
housing_data = housing_data.drop(['furnishingstatus'],axis=1)

min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0, 1)) 
housing_data.iloc[:,:] = min_max_scaler.fit_transform(housing_data.iloc[:,:]) 

X = housing_data.iloc[:,1:]  
Y = housing_data.iloc[:,0]

x_train, x_test, y_train, y_test = train_test_split(X,Y,train_size=0.9,random_state=100)

lm = LinearRegression()

lm.fit(x_train,y_train)

y_new = lm.predict(x_test)

c = [i for i in range(1,len(y_new)+1,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_new, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Sales', fontsize=16)         

mse = mean_squared_error(y_test, y_new)
r_squared = r2_score(y_test, y_new)

print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)

coeff_df = pd.DataFrame(lm.coef_,x_test.columns,columns=['Coefficient'])
print(coeff_df)