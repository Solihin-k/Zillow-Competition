#!/usr/bin/python

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

import gc

# import helper functions
from helpers import get_train_dataset
from preprocess import preprocessor

X_train, y_train = get_train_dataset('train_2016_v2', 'properties_2016')

# columns to drop
drop_cols = ['censustractandblock', 'rawcensustractandblock', 'fips', 
             'airconditioningtypeid', 'parcelid', 'propertylandusetypeid', 
             'regionidcounty', 'poolsizesum', 'regionidcity', 
             'unitcnt', 'architecturalstyletypeid', 'regionidneighborhood', 
             'regionidzip', 'buildingqualitytypeid', 'propertycountylandusecode', 
             'propertyzoningdesc', 'poolcnt', 'pooltypeid10', 'pooltypeid2', 
             'pooltypeid7', 'storytypeid', 'assessmentyear', 'decktypeid', 
             'buildingclasstypeid']

# preprocess data
p = preprocessor(cols_to_drop = drop_cols)

# create model
gbm = GradientBoostingRegressor(n_estimators = 1000, max_depth = 5, 
                                min_samples_split = 3, max_features = 15, 
                                learning_rate = 0.01, loss = 'lad', subsample = 0.8)

my_model = Pipeline([('preprocessor', p), ('regressor', gbm)])

# fit model
my_model.fit(X_train, y_train)

def make_chunks(df, chunksize = 50000):

    """ Generator to return chunks of a dataframe of a given size """

    chunk = 1
    total = len(df)//chunksize + 1

    while chunk <= total:
        if chunk < total:
            yield df.iloc[((chunk-1)*chunksize):(chunk*chunksize)]
        else:
            yield df.iloc[((chunk-1)*chunksize):]
        chunk += 1


def add_date(df, dt):

	""" Add prediction dates to original dataframe """

	df['transactiondate'] = pd.to_datetime(dt)
	return df

def make_sub_file(model, chunksize = 50000):

 	""" Create submission file for Kaggle """

 	dates = ['2016-10-01', '2016-11-01', '2016-12-01', '2017-10-01', '2017-11-01', '2017-12-01']
 	props = pd.read_csv('data/properties_2016.csv')
 	submission_df = pd.DataFrame(index=props.parcelid)

 	for d in dates:
 		props = add_date(props, d)

 		for x in make_chunks(props, chunksize):
 			preds = model.predict(x)
 			ix = x.parcelid
 			submission_df.loc[ix,str(pd.to_datetime(d).year) + str(pd.to_datetime(d).month)] = preds

 		print('processed date {0}'.format(d))

 	del props

 	return submission_df.round(4).reset_index()

make_sub_file(my_model, chunksize = 50000).to_csv('model_submission.csv', index = False)

# because of memory issues, garbage collect
gc.collect()