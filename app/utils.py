# (Definisi Custom Transformer untuk resolusi namespace joblib)

import pandas as pd
import numpy as np
# Wajib untuk semua Custom Transformer
from sklearn.base import BaseEstimator, TransformerMixin 


class CustomImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy.loc[(X_copy['age']>60) & (X_copy['job']=='unknown'), 'job'] = 'retired'
        X_copy.loc[(X_copy['education']=='unknown') & (X_copy['job']=='management'), 'education'] = 'university.degree'
        X_copy.loc[(X_copy['education']=='unknown') & (X_copy['job']=='services'), 'education'] = 'high.school'
        X_copy.loc[(X_copy['education']=='unknown') & (X_copy['job']=='housemaid'), 'education'] = 'basic.4y'
        basic_ed = ['basic.4y', 'basic.6y', 'basic.9y']
        X_copy.loc[(X_copy['job'] == 'unknown') & (X_copy['education'].isin(basic_ed)), 'job'] = 'blue-collar'
        X_copy.loc[(X_copy['job']=='unknown') & (X_copy['education']=='professional.course'), 'job'] = 'technician'
        return X_copy


class CyclicalFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_copy = X.copy()
        month_map = {'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
        day_map = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5}

        # Month
        if 'month' in X_copy.columns:
            X_copy['month_num'] = X_copy['month'].map(month_map)
            X_copy['month_sin'] = np.sin(2 * np.pi * X_copy['month_num']/12)
            X_copy['month_cos'] = np.cos(2 * np.pi * X_copy['month_num']/12)
            # errors='ignore' digunakan jika kolom sudah terhapus (misal: Month sudah diproses)
            X_copy.drop(columns=['month', 'month_num'], inplace=True, errors='ignore') 

        # Day
        if 'day_of_week' in X_copy.columns:
            X_copy['day_num'] = X_copy['day_of_week'].map(day_map)
            X_copy['day_sin'] = np.sin(2 * np.pi * X_copy['day_num']/5)
            X_copy['day_cos'] = np.cos(2 * np.pi * X_copy['day_num']/5)
            X_copy.drop(columns=['day_of_week', 'day_num'], inplace=True, errors='ignore')
            
        return X_copy
