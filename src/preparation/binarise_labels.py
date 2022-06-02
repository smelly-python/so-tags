"""
Module to binarize the labels.
"""
from os import path
import pickle as pkl
from sklearn.preprocessing import MultiLabelBinarizer


class Binarizer:
    """
        Wrapper for the multi label binarizer that adds some utility functions.
    """

    FILE_NAME = 'binarizer.pkl'

    def __init__(self, y_train):
        if y_train is not None:
            self.mlb = MultiLabelBinarizer(classes=sorted(get_tags_count(y_train).keys()))

    def binarize_training(self, y_train, y_val):
        """
        Creates a binary from the training data.
        """
        if not self.mlb:
            raise RuntimeError('mlb is uninitialised.'
                               'The Binarizer should not be created without providing y_train')
        # Just transform on y_val as it's the validation set
        return self.mlb.fit_transform(y_train), self.mlb.transform(y_val)

    def to_label(self, output):
        """
            output: the one hot encoded output of the classifier
            return a list of tags that correspond with it
        """
        return self.mlb.inverse_transform(output)

    def write_to_file(self, out_folder):
        """
            out_folder: folder in which the file should be written
            Writes the multi label binarizer to a pickle file
        """
        with open(path.join(out_folder, Binarizer.FILE_NAME), 'wb') as out_file:
            pkl.dump(self.mlb, out_file)

    @staticmethod
    def load_from_file(folder):
        """
            Reads the multi label binarizer from a file and creates a Binarizer with it

            folder: folder from which the file should be read
            return the Binarizer
        """
        with open(path.join(folder, Binarizer.FILE_NAME), 'rb') as in_file:
            binarizer = Binarizer(None)
            binarizer.mlb = pkl.load(in_file)
            return binarizer


def get_tags_count(data):
    """
    Counts the amount of tags
    """
    # Dictionary of all tags from train corpus with their counts.
    tags_counts = {}

    for tags in data:
        for tag in tags:
            if tag in tags_counts:
                tags_counts[tag] += 1
            else:
                tags_counts[tag] = 1

    return tags_counts
