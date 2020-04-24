#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import warnings
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# In[3]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


# In[4]:


from sklearn.metrics import make_scorer

def rmsle(actual_values, predicted_values):
    # 넘파이로 배열 형태로 바꿔준다.
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)
    
    # 예측값과 실제 값에 1을 더하고 로그를 씌워준다.
    log_predict = np.log(predicted_values + 1)
    log_actual = np.log(actual_values + 1)
    
    # 위에서 계산한 예측값에서 실제값을 빼주고 제곱을 해준다.
    difference = log_predict - log_actual
    # difference = (log_predict - log_actual) ** 2
    difference = np.square(difference)
    
    # 평균을 낸다.
    mean_difference = difference.mean()
    
    # 다시 루트를 씌운다.
    score = np.sqrt(mean_difference)
    
    return score

rmsle_scorer = make_scorer(rmsle)


# In[5]:


from sklearn.metrics import make_scorer

def neg_rmsle(actual_values, predicted_values):
    # 넘파이로 배열 형태로 바꿔준다.
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)
    
    # 예측값과 실제 값에 1을 더하고 로그를 씌워준다.
    log_predict = np.log(predicted_values + 1)
    log_actual = np.log(actual_values + 1)
    
    # 위에서 계산한 예측값에서 실제값을 빼주고 제곱을 해준다.
    difference = log_predict - log_actual
    # difference = (log_predict - log_actual) ** 2
    difference = np.square(difference)
    
    # 평균을 낸다.
    mean_difference = difference.mean()
    
    # 다시 루트를 씌운다.
    score = np.sqrt(mean_difference) * (-1)
    
    return score

neg_rmsle_scorer = make_scorer(neg_rmsle)


# In[6]:


class change_to_datetime(BaseEstimator, TransformerMixin):
    def __init__(self, column_name=None):
        self.column_name = column_name
        
    def fit(self, df_X, y=None):
        return self
    
    def transform(self, df_X):
        df_X[self.column_name] = pd.to_datetime(df_X[self.column_name])
        return df_X


# In[7]:


class change_to_str(BaseEstimator, TransformerMixin):
    def __init__(self, column_name=None):
        self.column_name = column_name
        
    def fit(self, df_X, y=None):
        return self
    
    def transform(self, df_X):
        df_X[self.column_name] = df_X[self.column_name].astype('str')
        return df_X


# In[8]:


class divide_datetime(BaseEstimator, TransformerMixin):       
    def fit(self, df_X, y=None):
        return self
    
    def transform(self, df_X):
        df_X["year"] = df_X["datetime"].dt.year
        df_X["month"] = df_X["datetime"].dt.month
        df_X["day"] = df_X["datetime"].dt.day
        df_X["hour"] = df_X["datetime"].dt.hour
        return df_X


# In[9]:


class add_dayofweek(BaseEstimator, TransformerMixin):       
    def fit(self, df_X, y=None):
        return self
    
    def transform(self, df_X):
        df_X["dayofweek"] = df_X["datetime"].dt.dayofweek
        return df_X


# In[10]:


def divide_columns(df_X) :
    columns = list(df_X.columns)

    num_columns = []
    cat_columns = []

    for column in columns :
        if df_X[column].dtypes == 'object' :
            cat_columns.append(column)

        else :
            num_columns.append(column)

    return num_columns, cat_columns


# In[11]:


class feature_selection(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        
    def fit(self, df_X, y=None):
        return self
    
    def transform(self, df_X):
        return df_X[self.columns]


# In[12]:


class one_hot_encoding(BaseEstimator, TransformerMixin):
    def __init__(self, column_name=None, prefix=None):
        self.column_name = column_name
        self.prefix = prefix
        
    def fit(self, df_X, y=None):
        return self
    
    def transform(self, df_X):
        onehotencoding = pd.get_dummies(df_X[self.column_name], prefix = self.prefix)
        df_X.drop(self.column_name, axis=1, inplace=True)
        return pd.concat([df_X, onehotencoding], axis=1)


# In[13]:


class standard_scaler(BaseEstimator, TransformerMixin):      
    def fit(self, df_X, y=None):
        return self
    
    def transform(self, df_X, y=None):
        scaler = StandardScaler()
        scaler.fit(df_X)
        X = scaler.transform(df_X)
        df_X = pd.DataFrame(X, columns=df_X.columns, index=df_X.index)
        return df_X


# In[14]:


class simple_imputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        
    def fit(self, df_X, y=None):
        return self
    
    def transform(self, df_X):
        imputer = SimpleImputer(strategy=self.strategy)
        np_X = imputer.fit_transform(df_X)
        df_X = pd.DataFrame(np_X, columns=df_X.columns, index=df_X.index)
        
        return df_X


# In[15]:


def fill_columns(df, columns, strategy='constant', value=0) :
    
    if strategy == 'constant' :
        df[columns].fillna(value, inplace=True)
        
    elif strategy == 'mean' :
        value = df[columns].mean()
        df[columns].fillna(value, inplace=True)
        
    elif strategy == 'median' :
        value = df[columns].median()
        df[columns].fillna(value, inplace=True)
        
    return df


# In[16]:


def rf_imputer(df, column_imp, columns_rf) :
    
    df_impute = df[df[column_imp].isnull()]
    df_rf = df[df[column_imp].notnull()]
    
#     df_rf[column_imp] = df_rf[column_imp].astype('str')
    
    rf_imp = RandomForestClassifier()
    rf_imp.fit(df_rf[columns_rf], df_rf[column_imp])
    impute_values = rf_imp.predict(df_impute[columns_rf])
    df_impute[column_imp] = impute_values
    
    df_new = pd.concat([df_impute, df_rf], axis=0)
    
#     df_new[column_imp] = df_new[column_imp].astype('float')
    
    return df_new


# In[17]:


def concat(df_A, df_B) : 
    return pd.concat([df_A, df_B], axis=1)


# In[18]:


def concat_list(df_list) : 
    return pd.concat(df_list, axis=1)


# In[19]:


class drop_feature(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        
    def fit(self, df_X, y=None):
        return self
    
    def transform(self, df_X):
        df_X.drop(self.columns, axis=1, inplace=True)
        return df_X

