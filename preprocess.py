#!/usr/bin/python

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class preprocessor:
    
    def __init__(self, cols_to_drop = None):
        
        self.cols_to_drop = cols_to_drop
        self.fitted = False
    
    
    def fit(self, features, y = None):
        """learn any information from the training data we may need to transform the test data"""
        
        
        self.fitted = True
        
            
        # drop variables
        features = features.drop(self.cols_to_drop, axis = 1)
        
        
        return self
    
    
    def transform(self, features):
        """transform the training or test data"""
        # transform the training or test data based on class attributes learned in the `fit` step
        
        if not self.fitted:
            raise NameError('Need to fit preprocessor first')

        # fill null values
        features = features.fillna(-1)
        
        # add col has_pool
        features['has_pool'] = ['Yes' if i == 1 else 'No' for i in features['poolcnt']]
        
        features['hashottuborspa'].fillna('False', inplace = True)
        features['fireplaceflag'].fillna('False', inplace = True)
        features['taxdelinquencyflag'].fillna('N', inplace = True)

        features['prop_age'] = 2020 - features['yearbuilt']
        features['total_rm_cnt'] = features['bathroomcnt'] + features['bedroomcnt']
        features['structure_value_psf'] = features['structuretaxvaluedollarcnt'] / features['calculatedfinishedsquarefeet']
        features['land_value_psf'] = features['landtaxvaluedollarcnt'] / features['calculatedfinishedsquarefeet']
        features['tax_value_psf'] = features['taxvaluedollarcnt'] / features['calculatedfinishedsquarefeet']

        # drop variables
        features = features.drop(self.cols_to_drop, axis = 1)

        # categorical features
        categorical_features = features.dtypes[features.dtypes == 'object'].index
        self.categorical_features = [x for x in categorical_features if 'date' not in x]
        
        # numerical features
        self.numerical_features = features.dtypes[features.dtypes == 'float64'].index
        
        # one-hot encoding
        features = pd.get_dummies(features, columns = self.categorical_features)
        
        # standardization
        scaled_features = features[self.numerical_features]
        scaler = StandardScaler().fit(scaled_features.values)
        scaled_features = scaler.transform(scaled_features.values)
        features[self.numerical_features] = scaled_features

		# convert transaction date column
        features['transactiondate'] = pd.to_datetime(features['transactiondate'])
        features['transaction_year'] = features['transactiondate'].dt.year
        features['transaction_month'] = features['transactiondate'].dt.month
        features['transaction_day'] = features['transactiondate'].dt.day

        features['transaction_day_sin'] = np.sin(features.transaction_day*(2.*np.pi/24))
        features['transaction_day_cos'] = np.cos(features.transaction_day*(2.*np.pi/24))
        features['transaction_month_sin'] = np.sin((features.transaction_month-1)*(2.*np.pi/12))
        features['transaction_month_cos'] = np.cos((features.transaction_month-1)*(2.*np.pi/12))

        features.drop(columns = ['transactiondate', 'transaction_month', 'transaction_day'], inplace = True)
        
        
        return features
    
    
    def fit_transform(self, features, y = None):
        """fit and transform wrapper method, used for sklearn pipeline"""

        return self.fit(features).transform(features)