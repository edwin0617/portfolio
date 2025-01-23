import numpy as np
import pandas as pd
import sklearn.metrics


def print_confusion_matrix(matrix, classes):
    max_name = max(len(c) for c in classes)
    print(" " * max_name + "True Class".center(len(classes) + max_name * (len(classes))))
    print("Predict".ljust(8) + " ".join([c.rjust(8) for c in classes]))
    for row, c in zip(matrix, classes):
        row = c.ljust(8) + " ".join([f"{i:8d}" for i in row])
        print(row)


def compute_cross_val_weighted_murmur_accuracy(val_predictions, print=True):
    df = pd.DataFrame.from_dict(val_predictions, orient="index")

    # Define order of classes in confusion matrix (and corresponding score weights)
    class_order = ["Present", "Unknown", "Absent"]
    matrix_weights = np.asarray([
        [5, 3, 1],
        [5, 3, 1],
        [5, 3, 1]
    ]).astype(float)

    conf_matrix = sklearn.metrics.confusion_matrix(df.label, df.prediction, labels=class_order).T
    
    if print:
        print_confusion_matrix(conf_matrix, class_order)
    weighted_conf = matrix_weights * conf_matrix
    if print:
        print_confusion_matrix(weighted_conf.astype(int), class_order)

    return np.trace(weighted_conf) / np.sum(weighted_conf)


def decide_murmur_with_threshold(row, threshold):
    murmur_pred = row['holo_HSMM']    
    
    if murmur_pred > threshold:
        return "Present"
    else:
        return "Absent"