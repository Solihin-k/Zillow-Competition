#!/usr/bin/python

import pandas as pd

def check_duplicates(df):
    
    """
    Check if a property has more than one transaction.
    
    Keyword arguments:
    - df -> the merged train dataset
    
    Returns: 
    - a dataframe with properties containing multiple transactions
    - a dataframe with properties containing only one transaction
    
    """
    
    parcel_counts = df.groupby('parcelid').size()
    multiple_sales = df[df.parcelid.isin(parcel_counts[parcel_counts > 1].index)]
    single_sale = df[df.parcelid.isin(parcel_counts[parcel_counts == 1].index)]

    
    return multiple_sales, single_sale


def filter_duplicates(df, random_state = 0):
    
    """
    Filter the merged train dataset to only include one record per parcel.

    Keyword arguments:
    df -> the merged train dataset
    random_state -> the random seed to be passed to the `pandas.DataFrame.sample()` method
    
    Returns:
    - a dataframe containing only one transaction per property
    
    """

    #reduced_df = duplicates.sample(frac = 1, random_state = random_state).groupby('parcelid').head(1)
    #reduced_df = pd.concat([non_duplicates, reduced_df])

    counts_per_parcel = df.groupby('parcelid').size()
    more_than_one_sale = df[df.parcelid.isin(counts_per_parcel[counts_per_parcel > 1].index)]
    only_one_sale = df[df.parcelid.isin(counts_per_parcel[counts_per_parcel == 1].index)]
    reduced_df = more_than_one_sale.sample(frac=1, random_state=random_state).groupby('parcelid').head(1)
    reduced_df = pd.concat([only_one_sale, reduced_df])
    
    return reduced_df


def get_train_dataset(train_file, properties_file):
    
    """Create the training dataset (2016) or the test dataset (2017)

    Keyword arguments:
    dset -- a string in {train, test}
    
    Returns:
    a tuple of pandas dataframe (X) and pandas series (y)
    """
    
    train = pd.read_csv('data/{}.csv'.format(train_file))
    properties = pd.read_csv('data/{}.csv'.format(properties_file))
    merged = pd.merge(train, properties, how = 'left', on = 'parcelid')
    
    merged = filter_duplicates(merged)
    
    y = merged.pop('logerror')
    return merged, y