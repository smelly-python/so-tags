import src.settings as settings
import re
import numpy as np
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer

DICT_SIZE = 5000


def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = re.sub(settings.REPLACE_BY_SPACE_RE, " ", text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(settings.BAD_SYMBOLS_RE, "", text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = " ".join([word for word in text.split() if not word in settings.STOPWORDS])  # delete stopwords from text
    return text


def data_text_prepare(x):
    return [text_prepare(x) for x in x]


def get_words_count(x):
    # Dictionary of all words from train corpus with their counts.
    words_counts = {}

    for sentence in x:
        for word in sentence.split():
            if word in words_counts:
                words_counts[word] += 1
            else:
                words_counts[word] = 1

    return sorted(words_counts, key=words_counts.get, reverse=True)


def create_bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary

        return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)

    for word in text.split():
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1
    return result_vector


def get_bag(x):
    index_to_words = get_words_count(x)[:DICT_SIZE]
    words_to_index = {word: i for i, word in enumerate(index_to_words)}
    return sp_sparse.vstack([sp_sparse.csr_matrix(create_bag_of_words(text, words_to_index, DICT_SIZE)) for text in x])


def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern='(\S+)')

    # Fit the vectorizer on the train set
    train_res = tfidf_vectorizer.fit_transform(X_train)

    # Transform the train, test, and val sets and return the result
    val_res = tfidf_vectorizer.transform(X_val)
    test_res = tfidf_vectorizer.transform(X_test)

    return train_res, val_res, test_res, tfidf_vectorizer.vocabulary_
