"""
Module responsible for creating the dataset.
"""
from ast import literal_eval
import pandas as pd


def read_data(filename):
    """
    Read the data.
    """
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data


def make_dataset():
    """
    Create a dataset
    """
    train = read_data('data/train.tsv')
    x_train, y_train = train['title'].values, train['tags'].values

    validation = read_data('data/validation.tsv')
    x_val, y_val = validation['title'].values, validation['tags'].values

    test = pd.read_csv('data/test.tsv', sep='\t')
    x_test = test['title'].values

    return x_train, y_train, x_val, y_val, X_test
