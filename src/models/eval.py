import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def classification_eval(logits, labels, threshold = 0.5, pad_mask=None):

    # Remove indices with pad tokens
    input_probs = softmax(logits, axis=1)
    classes = (input_probs[:, -1] >= threshold)
    
    results = {}
    results["classification_precision"] = precision_score(
        labels, classes)
    results["classification_recall"] = recall_score(labels, classes)
    results["classification_accuracy"] = accuracy_score(
        labels, classes)
    results["classification_f1"] = f1_score(labels, classes)
    results["roc_auc"] = roc_auc_score(labels,input_probs[:,-1])

    return results
