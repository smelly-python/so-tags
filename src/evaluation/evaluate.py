import os
import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score


def write_evaluation_scores(res_dir, y_val, predicted):
    with open(res_dir + "scores.txt", "w") as f:
        f.write('Accuracy score: ' + str(accuracy_score(y_val, predicted)))
        f.write('F1 score: ' + str(f1_score(y_val, predicted, average='weighted')))
        f.write('Average precision score: ' + str(average_precision_score(y_val, predicted, average='macro')))
        f.close()


def evaluate(y_val, predicted):
    now = time.strftime("%d-%m-%Y_%H-%M-%S")
    res_dir = "reports/results_" + now + "/"
    os.mkdir(res_dir)

    write_evaluation_scores(res_dir, y_val, predicted)
