"""
Module responsible for using the model to predict samples
"""

from src.preparation.binarise_labels import Binarizer
from src.preparation.build_features import Vectorizer, text_prepare


class Predictor:

    def __init__(self, classifier, vectorizer: Vectorizer, binarizer: Binarizer):
        self.clf = classifier
        self.vectorizer = vectorizer
        self.binarizer = binarizer

    def predict_sample(self, sample):
        features = self.vectorizer.featurize(text_prepare(sample))
        result = self.clf.predict(features)
        return self.binarizer.to_label(result)
