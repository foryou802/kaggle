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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib


# In[3]:


class change_to_datetime(BaseEstimator, TransformerMixin):
    def __init__(self, column_name=None):
        self.column_name = column_name
        
    def fit(self, df_X, y=None):
        return self
    
    def transform(self, df_X):
        df_X[self.column_name] = pd.to_datetime(df_X[self.column_name])
        return df_X


# In[4]:


class divide_datetime(BaseEstimator, TransformerMixin):       
    def fit(self, df_X, y=None):
        return self
    
    def transform(self, df_X):
        df_X["year"] = df_X["datetime"].dt.year
        df_X["month"] = df_X["datetime"].dt.month
        df_X["day"] = df_X["datetime"].dt.day
        df_X["hour"] = df_X["datetime"].dt.hour
        return df_X


# In[5]:


class add_dayofweek(BaseEstimator, TransformerMixin):       
    def fit(self, df_X, y=None):
        return self
    
    def transform(self, df_X):
        df_X["dayofweek"] = df_X["datetime"].dt.dayofweek
        return df_X


# In[6]:


class feature_selection(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        
    def fit(self, df_X, y=None):
        return self
    
    def transform(self, df_X):
        return df_X[self.columns]


# In[7]:


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


# In[8]:


class standard_scaler(BaseEstimator, TransformerMixin):      
    def fit(self, df_X, y=None):
        return self
    
    def transform(self, df_X, y=None):
        scaler = StandardScaler()
        scaler.fit(df_X)
        X = scaler.transform(df_X)
        df_X = pd.DataFrame(X, columns=df_X.columns, index=df_X.index)
        return df_X


# In[9]:


def concat(df_A, df_B) : 
    return pd.concat([df_A, df_B], axis=1)


# In[10]:


class drop_feature(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        
    def fit(self, df_X, y=None):
        return self
    
    def transform(self, df_X):
        df_X.drop(self.columns, axis=1, inplace=True)
        return df_X

