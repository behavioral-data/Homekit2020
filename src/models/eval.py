import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score,precision_recall_fscore_support, roc_auc_score
from functools import partial

def classification_eval(logits, labels, threshold = 0.5):

    # Remove indices with pad tokens
    input_probs = softmax(logits, axis=1)
    classes = (input_probs[:, -1] >= threshold)
    precision, recall, f1, support = precision_recall_fscore_support(labels, classes, average='binary')
    results = {}
    results["classification_precision"] = precision
    results["classification_recall"] = recall
    results["classification_f1"] = f1
    results["classification_support"] = support

    accuracy =  accuracy_score(labels, classes)
    results["classification_accuracy"] = accuracy

    results["roc_auc"] = roc_auc_score(labels,input_probs[:,-1])

    return results

def get_huggingface_classification_eval(threshold=0.5):

    # Remove indices with pad tokens
    def evaluator(pred):
        labels = pred.label_ids
        logits = pred.predictions
        return classification_eval(logits,labels,threshold=threshold)
    
    return evaluator

