"""
Module responsible for training the model.
"""
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


def train_classifier(x_train, y_train, penalty='l1', const=1):
    """
      X_train, y_train — training data

      return: trained classifier
    """

    # Create and fit LogisticRegression wrapped into OneVsRestClassifier.

    clf = LogisticRegression(
        penalty=penalty,
        C=const,
        dual=False,
        solver='liblinear')
    clf = OneVsRestClassifier(clf)
    clf.fit(x_train, y_train)

    return clf
