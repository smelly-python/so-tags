"""
Module responsible for creating the model.
"""
from ast import literal_eval

import pandas as pd
import pickle as pkl
from os import path
from src.models.train_model import train_classifier
from src.data.make_dataset import make_dataset
from src.preparation.binarise_labels import Binarizer
from src.preparation.build_features import Vectorizer, data_text_prepare


def read_data(filename):
    """
    Read data from a file.
    """
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data


if __name__ == '__main__':
    # TODO: refactor into class that just takes train set
    # make dataset
    X_train, y_train, X_val, y_val, X_test = make_dataset()
    X_train = data_text_prepare(X_train)
    X_val = data_text_prepare(X_val)
    X_test = data_text_prepare(X_test)
    print(X_train)
    # build preparation
    vectorizer = Vectorizer()
    X_train, X_val, X_test, vocab = vectorizer.tfidf_features_training(X_train, X_val, X_test)
    print(X_train)
    vectorizer.write_to_file('output')

    # binarize labels
    binarizer = Binarizer(y_train)
    y_train, y_val = binarizer.binarize_training(y_train, y_val)
    binarizer.write_to_file('output')
    print(y_train)

    # train model
    clf = train_classifier(X_train, y_train)

    with open(path.join('output', 'classifier.pkl'), 'wb') as out_file:
        pkl.dump(clf, out_file)
        print("Stored the model in output/classifier.pkl")
