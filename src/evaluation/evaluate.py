"""
Module responsible for the evaluation of the model.
"""
import json

from os import path
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score


class EvaluationResult:
    """
    Holds the metrics computed during the evaluation
    """

    def __init__(self, accuracy, f_1, precision):
        self.accuracy_score = accuracy
        self.f1_score = f_1
        self.average_precision_score = precision

    def to_json(self):
        """
        Returns the evaluation object as a json string.
        :return: the json string
        """
        return json.dumps(self.__dict__)

    def write_evaluation_scores(self, res_dir: str):
        """
        Writes the evaluation scores to a file.
        """
        with open(path.join(res_dir, 'evaluation.json'), "w", encoding='utf-8') as file:
            file.write(self.to_json())
            file.close()


def evaluate(y_val, predicted, out_folder):
    """
    Performs the evaluation on the data.
    """
    acc = accuracy_score(y_val, predicted)
    f_1 = f1_score(y_val, predicted, average='weighted')
    precision = average_precision_score(y_val, predicted, average='macro')
    eval_result = EvaluationResult(acc, f_1, precision)
    eval_result.write_evaluation_scores(out_folder)
