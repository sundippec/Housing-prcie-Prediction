# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:15:23 2019

@author: sandeep
"""

#we are importing required libraries

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import LinearSVR, SVR
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


#loading the dataset

train = pd.read_csv('dataset.csv')
train.head()
train.shape
train.info()
train.columns[train.isnull().any()]


#Normal distribution
plt.subplots(figsize=(12,9))
sns.distplot(train['SalePrice'], fit=stats.norm)

# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(train['SalePrice'])

# plot with the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')

#Probablity plot

fig = plt.figure()
stats.probplot(train['SalePrice'], plot=plt)
plt.show()

#check for null values

train.columns[train.isnull().any()]

#HeatMap
plt.figure(figsize=(12, 6))
sns.heatmap(train.isnull())
plt.show()
#count of missing values
Isnull = train.isnull().sum()/len(train)*100
Isnull = Isnull[Isnull>0]
Isnull.sort_values(inplace=True, ascending=False)
Isnull = Isnull.to_frame()
Isnull.columns = ['count']
Isnull.index.names = ['Name']
Isnull['Name'] = Isnull.index

#corelation 

train_corr = train.select_dtypes(include=[np.number])
train_corr.shape
del train_corr['Id']
#Corelation
col = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
#heatMap
corr = train_corr.corr()
plt.subplots(figsize=(20,9))
sns.heatmap(corr, annot=True)


#important attributes
top_feature = corr.index[abs(corr['SalePrice']>0.5)]
plt.subplots(figsize=(12, 8))
top_corr = train[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

#unique value of OverallQual
train.OverallQual.unique()
sns.barplot(train.OverallQual, train.SalePrice)

#boxplot
plt.figure(figsize=(18, 8))
sns.boxplot(x=train.OverallQual, y=train.SalePrice)
col = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
sns.set(style='ticks')
sns.pairplot(train[col], size=3, kind='reg')


print("Important attributes relative to target")
corr = train.corr()
corr.sort_values(['SalePrice'], ascending=False, inplace=True)
corr.SalePrice

#training and cleaning
train['PoolQC'] = train['PoolQC'].fillna('None')
train['MiscFeature'] = train['MiscFeature'].fillna('None')
train['Alley'] = train['Alley'].fillna('None')
train['Fence'] = train['Fence'].fillna('None')
train['FireplaceQu'] = train['FireplaceQu'].fillna('None')
train['LotFrontage'] = train.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    train[col] = train[col].fillna('None')
    
for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
    train[col] = train[col].replace('?' ,'1980' )

 
    
for col in ('BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'):
    train[col] = train[col].fillna('None')
    
    train['MasVnrArea'] = train['MasVnrArea'].fillna(int(0))
    
    train['MasVnrType'] = train['MasVnrType'].fillna('None')
    train['Electrical'] = train['Electrical'].fillna(train['Electrical']).mode()[0]
    train = train.drop(['Utilities'], axis=1)
    
    #converting string to int
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'GarageType', 'MiscFeature', 
        'SaleType', 'SaleCondition', 'Electrical', 'Heating')
    
from sklearn.preprocessing import LabelEncoder
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train[c].values)) 
    train[c] = lbl.transform(list(train[c].values))
    
    
    #dependent variable to y
    y = train['SalePrice']
    
    del train['SalePrice']
    X = train.values
    y = y.values
    
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

#Train the model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000)

model.fit(X_train, y_train)

 
print("Predict value " + str(model.predict(X_test)))
print("Real value " + str(y_test))

y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("RMSE: %f" % (rmse))

print("Accuracy --> ", model.score(X_test, y_test)*100)

#Differentmodels
from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("RMSE: %f" % (rmse))

print("Accuracy --> ", model.score(X_test, y_test)*100)

train.to_csv("cleaned.csv")

import lightgbm as lgb
model = lgb.LGBMRegressor(num_leaves=10, n_estimators=100, nthread=4, colsample_bytree=0.8, subsample=0.8)

model.fit(X_train, y_train)

print("Predict value " + str(model.predict(X_test)))
print("Real value " + str(y_test))

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("RMSE: %f" % (rmse))
print("Accuracy --> ", model.score(X_test, y_test)*100)

#svr

from sklearn.svm import LinearSVR, SVR
model = SVR(C=500)
model.fit(X_train, y_train)

print("Predict value " + str(model.predict(X_test)))
print("Real value " + str(y_test))

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("RMSE: %f" % (rmse))

print("Accuracy --> ", model.score(X_test, y_test)*100)
#MLP regressor

from sklearn.neural_network import MLPRegressor

model = MLPRegressor(hidden_layer_sizes=(32, 16), random_state=15)
model.fit(X_train, y_train)

print("Predict value " + str(model.predict(X_test)))
print("Real value " + str(y_test))

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("RMSE: %f" % (rmse))


print("Accuracy --> ", model.score(X_test, y_test)*100)

