"""
Module responsible for the evaluation of the model.
"""
import json
import os
import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score


class EvaluationResult:

    def __init__(self, accuracy, f1, precision):
        self.accuracy_score = accuracy
        self.f1_score = f1
        self.average_precision_score = precision

    def to_json(self):
        return json.dumps(self.__dict__)


def write_evaluation_scores(res_dir: str, result: EvaluationResult):
    """
    Writes the evaluation scores to a file.
    """
    with open(res_dir + "scores.txt", "w") as file:
        file.write(result.to_json())
        file.close()


def evaluate(y_val, predicted):
    """
    Performs the evaluation on the data.
    """
    now = time.strftime("%d-%m-%Y_%H-%M-%S")
    res_dir = "reports/results_" + now + "/"
    os.mkdir(res_dir)

    acc = accuracy_score(y_val, predicted)
    f1 = f1_score(y_val, predicted, average='weighted')
    precision = average_precision_score(y_val, predicted, average='macro')

    write_evaluation_scores(res_dir, EvaluationResult(acc, f1, precision))
