"""
Module responsible for creating the model.
"""
from ast import literal_eval
from os import path
import pickle as pkl
import pandas as pd

from src.evaluation.evaluate import evaluate
from src.models.train_model import train_classifier
from src.data.make_dataset import make_dataset
from src.preparation.binarise_labels import Binarizer
from src.preparation.build_features import Vectorizer, data_text_prepare


def read_data(filename):
    """
    Read data from a file.
    """
    data = pd.read_csv(filename, sep='\t', dtype={'title': 'str', 'tags': 'str'})[['title', 'tags']]
    data['tags'] = data['tags'].apply(literal_eval)
    return data


if __name__ == '__main__':

    OUTPUT_FOLDER = 'output'

    # make dataset
    X_train, y_train, X_val, y_val, X_test = make_dataset()
    X_train = data_text_prepare(X_train)
    X_val = data_text_prepare(X_val)
    X_test = data_text_prepare(X_test)
    # build preparation
    vectorizer = Vectorizer()
    X_train, X_val, X_test, vocab = vectorizer.tfidf_features_training(X_train, X_val, X_test)
    vectorizer.write_to_file(OUTPUT_FOLDER)

    # binarize labels
    binarizer = Binarizer(y_train)
    y_train, y_val = binarizer.binarize_training(y_train, y_val)
    binarizer.write_to_file(OUTPUT_FOLDER)

    # train model
    clf = train_classifier(X_train, y_train)

    # Evaluate
    y_val_predicted_labels_tfidf = clf.predict(X_val)
    y_val_predicted_scores_tfidf = clf.decision_function(X_val)

    # evaluation
    evaluate(y_val, y_val_predicted_labels_tfidf, OUTPUT_FOLDER)

    with open(path.join(OUTPUT_FOLDER, 'classifier.pkl'), 'wb') as out_file:
        pkl.dump(clf, out_file)
        print(f'Stored the model in {OUTPUT_FOLDER}/classifier.pkl')
