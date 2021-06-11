from os import rename
import numpy as np
import pandas as pd
import wandb
import copy
from wandb.viz import CustomChart
from wandb.data_types import Table

import torch
from torchmetrics import Metric

from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr

from sklearn.metrics import (accuracy_score,precision_recall_fscore_support, roc_auc_score,
                            mean_absolute_error)


from functools import partial
from src.utils import check_for_wandb_run

def classification_eval(logits, labels, threshold = 0.5, prefix=None):

    # Remove indices with pad tokens
    input_probs = softmax(logits, axis=1)
    classes = (input_probs[:, -1] >= threshold)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, classes, average='binary')
    results = {}
    results["classification_precision"] = precision
    results["classification_recall"] = recall
    results["classification_f1"] = f1
    
    support = pd.Series(labels).value_counts().to_dict()
    results["classification_support"] = support

    accuracy = accuracy_score(labels, classes)
    results["classification_accuracy"] = accuracy

    try:
        results["roc_auc"] = roc_auc_score(labels,input_probs[:,-1])
    except ValueError:
        results["roc_auc"] = np.nan

    if check_for_wandb_run():
        results["roc"] = wandb.plot.roc_curve(labels, input_probs,
                                            labels=["Negative","Positive"], classes_to_plot=[1])
        results["pr"] = wandb.plot.pr_curve(labels, input_probs,
    
                                          labels=["Negative","Positive"], classes_to_plot=[1])
    if prefix:
        renamed = {}
        for k,v in results.items():
            renamed[prefix+k] = v
        results = renamed
    
    return results

def get_huggingface_classification_eval(threshold=0.5):

    # Remove indices with pad tokens
    def evaluator(pred):
        labels = pred.label_ids
        logits = pred.predictions
        return classification_eval(logits,labels,threshold=threshold)
    
    return evaluator



def autoencode_eval(pred, labels):
    # Remove indices with pad tokens
    results = {}

    pred_flat = pred.flatten()
    labels_flat = labels.flatten()

    results["pearson"] = pearsonr(pred_flat,labels_flat)[0]
    results["spearman"] = spearmanr(pred_flat,labels_flat)[0]
    results["MAE"] = mean_absolute_error(pred_flat,labels_flat)

    return results