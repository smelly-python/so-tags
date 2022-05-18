from src.data.make_dataset import make_dataset
from src.preparation.build_features import data_text_prepare, tfidf_features
from src.preparation.binarise_labels import binarise
from src.models.train_model import train_classifier
from src.evaluation.evaluate import evaluate

if __name__ == '__main__':
    '''
    Main method for the src packages.   
    '''
    # make dataset
    X_train, y_train, X_val, y_val, X_test = make_dataset()

    # build preparation
    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = \
        tfidf_features(data_text_prepare(X_train),
                       data_text_prepare(X_val), data_text_prepare(X_test))
    tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}

    # binarise labels
    y_train, y_val = binarise(y_train, y_val)

    # train model
    classifier_tfidf = train_classifier(X_train_tfidf, y_train)
    # predict labels and scores
    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(
        X_val_tfidf)

    # evaluation
    evaluate(y_val, y_val_predicted_labels_tfidf)
