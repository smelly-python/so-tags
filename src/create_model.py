import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from models import train_model
from preparation import build_features
from preparation import binarise_labels
from ast import literal_eval

from joblib import dump, load

def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data

if __name__ == '__main__':
    train = read_data('./data/train.tsv')
    X_train, y_train = train['title'].values, train['tags'].values
    mlb = MultiLabelBinarizer(classes=sorted(binarise_labels.get_tags_count(y_train).keys()))
    X_train = [build_features.text_prepare(x) for x in X_train]
    y_train = mlb.fit_transform(y_train)
    X_train_mybag = build_features.get_bag(X_train)

    clf = train_model.train_classifier(X_train_mybag, y_train)

    dump(clf, 'output/model.joblib')
    print("Stored the model in output/model.joblib")