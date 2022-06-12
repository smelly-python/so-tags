"""
Module responsible for using the model to predict samples
"""

from src.preparation.binarise_labels import Binarizer
from src.preparation.build_features import Vectorizer, text_prepare


class Predictor:
    """
        Contains all the parts of the system to make predictions after training.
    """

    def __init__(self, classifier, vectorizer: Vectorizer,
                 binarizer: Binarizer):
        self.clf = classifier
        self.vectorizer = vectorizer
        self.binarizer = binarizer

    def predict_sample(self, sample):
        """
            Predicts a single sample using the featurizer, classifier, and binarizer.

            sample: the title to predict
            return a list of tags
        """
        features = self.vectorizer.featurize(text_prepare(sample))
        result = self.clf.predict(features)
        return self.binarizer.to_label(result)

    def predict_samples(self, samples):
        """
            Predicts multiple samples using the featurizer, classifier, and binarizer.

            samples: the titles to predict
            return a list with lists of tags
        """
        return [self.predict_sample(sample) for sample in samples]
