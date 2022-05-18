"""
Module responsible for the evaluation of the model.
"""
import os
import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score


def write_evaluation_scores(res_dir, y_val, predicted):
    """
    Writes the evaluation scores to a file.
    """
    with open(res_dir + "scores.txt", "w") as file:
        file.write('Accuracy score: ' + str(accuracy_score(y_val, predicted)))
        file.write('F1 score: ' +
                   str(f1_score(y_val, predicted, average='weighted')))
        file.write('Average precision score: ' +
                   str(average_precision_score(y_val, predicted, average='macro')))
        file.close()


def evaluate(y_val, predicted):
    """
    Performs the evaluation on the data.
    """
    now = time.strftime("%d-%m-%Y_%H-%M-%S")
    res_dir = "reports/results_" + now + "/"
    os.mkdir(res_dir)

    write_evaluation_scores(res_dir, y_val, predicted)
