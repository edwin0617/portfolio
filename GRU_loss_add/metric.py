import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

import sklearn.metrics
import sklearn.model_selection



def print_confusion_matrix(matrix, classes):
    max_name = max(len(c) for c in classes)
    print(" " * max_name + "True Class".center(len(classes) + max_name * (len(classes))))
    print("Predict".ljust(8) + " ".join([c.rjust(8) for c in classes]))
    for row, c in zip(matrix, classes):
        row = c.ljust(8) + " ".join([f"{i:8d}" for i in row])
        print(row)


def C_EXPERT(s, t):
    s_t = s / t
    return t * (25 + (397 * s_t) - (1718 * s_t ** 2) + (11296 * s_t **4))


def outcome_cost(tp, tn, fp, fn, printl=False):
    num_patients = tp + tn + fp + fn

    C_ALGORITHM = 10
    C_TREATMENT = 10_000
    C_ERROR = 50_000

    c_algo = C_ALGORITHM * num_patients
    c_exp = C_EXPERT(tp + fp, num_patients)
    c_treat = C_TREATMENT * tp
    c_err = C_ERROR * fn
    total_cost = c_algo + c_exp + c_treat + c_err
    result = total_cost / num_patients

    if printl:
        print(f"C_ALGO = {c_algo/num_patients:.0f}, C_EXP = {c_exp/num_patients:4.0f}, C_TREAT = {c_treat/num_patients:4.0f}, C_ERR = {c_err/num_patients:5.0f}, C_TOTAL = {result:5.0f}")

    return result



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

    # print(sklearn.metrics.classification_report(df.label, df.prediction, labels=class_order))

    # from evaluate_model import compute_auc
    # MAP = {"Present": 0, "Unknown": 1, "Absent": 2}
    # labels = np.zeros((len(df), 3))
    # for i in range(len(labels)):
    #     labels[i, MAP[df.iloc[i].label]] = 1 
    # output = np.stack(df.probabilities)
    # print(compute_auc(labels, output))

    return np.trace(weighted_conf) / np.sum(weighted_conf)


def compute_cross_val_outcome_score(val_predictions):
    df = pd.DataFrame.from_dict(val_predictions, orient="index")

    class_order = ["Abnormal", "Normal"]
    conf_matrix = sklearn.metrics.confusion_matrix(df.label, df.prediction, labels=class_order).T
    print_confusion_matrix(conf_matrix, class_order)

    tp, fp, fn, tn = conf_matrix.ravel()

    cost = outcome_cost(tp, tn, fp, fn)
    return cost




def compute_outcome_score(prediction, label, print_matrix=False):
    conf_matrix = sklearn.metrics.confusion_matrix(label, prediction, labels=[1, 0]).T
    tp, fp, fn, tn = conf_matrix.ravel()
    if print_matrix:
        print_confusion_matrix(conf_matrix, ["Abnormal", "Normal"])
    return outcome_cost(tp, tn, fp, fn)



def choose_outcome_threshold(df):
    df["first_prob"] = [p[0] for p in df["probabilities"]]
    df["label_int"] = df["label"].map({"Abnormal": 1, "Normal": 0})

    rows = []
    for threshold in sorted(df["first_prob"].unique())[:-1]:
        cost = compute_outcome_score(df["first_prob"] > threshold, df["label_int"])
        rows.append({"cost": cost, "threshold": threshold})
    rows_df = pd.DataFrame.from_records(rows)

    min_row = rows_df.iloc[rows_df.cost.idxmin()]
    compute_outcome_score(df["first_prob"] > min_row.threshold, df["label_int"], print_matrix=True)

    return min_row.cost, min_row.threshold


# def decide_murmur_outcome(features: Dict):
#     conf_differences = [v for k, v in features.items() if k.startswith("conf_difference_")]
#     signal_quals = [v for k, v in features.items() if k.startswith("signal_qual_")]

#     if np.nanmax(conf_differences) > 0:
#         return "Present"
#     elif np.nanmin(signal_quals) < 0.65:
#         return "Unknown"
#     else:
#         return "Absent"


def decide_murmur_outcome(features: Dict):
    conf_differences = [v for k, v in features.items() if k.startswith("holo_HSMM")]
    # signal_quals = [v for k, v in features.items() if k.startswith("signal_qual_")]


    #print(len(conf_differences))


    if np.mean(conf_differences) > 0.35:
        return "Present"
    # elif np.nanmin(signal_quals) < 0.65:
    #     return "Unknown"
    else:
        return "Absent"


    
def decide_murmur(row: Dict):
    murmur_pred = row['holo_HSMM']    
    
    if murmur_pred > 0.45:
        return "Present"
    #elif 
    else:
        return "Absent"
    
    
def decide_murmur_with_threshold(row, threshold):
    murmur_pred = row['holo_HSMM']    
    
    if murmur_pred > threshold:
        return "Present"
    #elif 
    else:
        return "Absent"