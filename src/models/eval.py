from os import rename
import numpy as np
from numpy.core.shape_base import vstack
import pandas as pd
import wandb
import copy
from wandb.plot.roc_curve import roc_curve
from wandb.viz import CustomChart
from wandb.data_types import Table

import torch
import torchmetrics
from torchmetrics import BinnedPrecisionRecallCurve

from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr

from sklearn.metrics import (accuracy_score,precision_recall_fscore_support, roc_auc_score,
                            mean_absolute_error, det_curve, precision_recall_curve, auc)


from functools import partial
from src.utils import check_for_wandb_run


def make_ci_bootsrapper(estimator):
    def bootstrapper(pred,labels,n_samples=100):
        results = []
        inds = np.arange(len(pred))
        for _ in range(n_samples):
            sample = np.random.choice(inds,len(pred))
            try:
                results.append(estimator(pred[sample],labels[sample]))
            except ValueError:
                continue
        return np.quantile(results,0.05), np.quantile(results,0.95)
    return bootstrapper

def get_wandb_plots():
    ...
def classification_eval(preds, labels, threshold = None, prefix=None,
                        bootstrap_cis=False):
    results = {}
    # Remove indices with pad tokens
    if threshold:
        input_probs = softmax(preds, axis=1)
        classes = (input_probs[:, -1] >= threshold)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, classes, average='binary')
        accuracy = accuracy_score(labels, classes)
        results["precision"] = precision
        results["recall"] = recall
        results["f1"] = f1
        results["accuracy"] = accuracy

    support = pd.Series(labels).value_counts().to_dict()
    results["support"] = support

    
    if bootstrap_cis:
        results["roc_auc"], (results["roc_auc_ci_low"] ,results["roc_auc_ci_high"]) = roc_auc(preds,labels,get_ci=True)
        results["pr_auc"], (results["pr_auc_ci_low"], results["pr_auc_ci_high"]) = pr_auc(preds,labels,get_ci=True)
    else:
        results["roc_auc"] = roc_auc(preds,labels,get_ci=False)
        results["pr_auc"] = pr_auc(preds,labels,get_ci=False)

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

def wandb_pr_curve(preds,labels):
    # preds = preds.cpu().numpy()
    # labels = labels.cpu().numpy()
    pr_curve = BinnedPrecisionRecallCurve(num_classes=1) 
    precision, recall, _thresholds = pr_curve(preds, labels)
    label_markers = ["Positive"] * len(precision)
    table = Table(columns= ["class","precision","recall"], data=list(zip(label_markers,precision,recall)))

    plot = wandb.plot_table(
        "wandb/precision-recall-curve/v0",
        table,
        {"x": "recall", "y": "precision", "class": "class"},
        {
            "title": "Precision Recall Curve",
            "x-axis-title": "Recall",
            "y-axis-title": "Precision",
        },

    )
    return plot
    

def wandb_detection_error_tradeoff_curve(preds,labels,return_table=False,limit=999):
    preds = preds.cpu().numpy()
    # probs = np.stack((1-preds,preds)).T
  
    fpr, fnr, _ = det_curve(labels, preds)
    if limit and limit < len(fpr):
        inds = np.random.choice(len(fpr), size=limit,replace=False)
        fpr = fpr[inds]
        fnr = fnr[inds]
        
    label_markers = ["Positive"] * len(fpr)
    table = Table(columns= ["class","fpr","fnr"], data=list(zip(label_markers,fpr,fnr)))
    if return_table:
        return table
    plot = wandb.plot_table(
        "wandb/error-tradeoff-curve/v0",
        table,
        {"x": "fpr", "y": "fnr", "class": "class"},
        {
            "title": "Error Tradeoff Curve",
            "x-axis-title": "False positive rate",
            "y-axis-title": "False negative rate",
        },

    )
    return plot

def wandb_roc_curve(preds,labels, return_table=False,limit=999):
    fpr, tpr, _ = torchmetrics.functional.roc(preds, labels, pos_label=1)
    if limit and limit < len(fpr):
        inds = np.random.choice(len(fpr), size=limit,replace=False)
        fpr = fpr[inds]
        tpr = tpr[inds]

    label_markers = ["Positive"] * len(fpr)
    table = Table(columns= ["class","fpr","tpr"], data=list(zip(label_markers,fpr,tpr)))
    if return_table:
        return table
    plot = wandb.plot_table(
        "wandb/area-under-curve/v0",
        table,
        {"x": "fpr", "y": "tpr", "class": "class"},
        {
            "title": "ROC",
            "x-axis-title": "False positive rate",
            "y-axis-title": "True positive rate",
        },

    )
    return plot

def pr_auc(pred,labels,get_ci=False,n_samples=10000):
    preds = pred.cpu().numpy()
    precision, recall, _ = precision_recall_curve(labels,preds)
    result = auc(recall,precision)
    if get_ci:
        ci = make_ci_bootsrapper(pr_auc)(pred,labels,n_samples=n_samples)
        return result, ci
    else:
        return result


def roc_auc(pred,labels,get_ci=False,n_samples=10000):
    preds = pred.cpu().numpy()
    score = roc_auc_score(labels,preds)
    if get_ci:
        ci = make_ci_bootsrapper(lambda x,y : roc_auc_score(y_score=x, y_true=y))(pred,labels,n_samples=n_samples)
        return score, ci
    else:
        return score
    
def autoencode_eval(pred, labels):
    # Remove indices with pad tokens
    results = {}

    pred_flat = pred.flatten()
    labels_flat = labels.flatten()

    results["pearson"] = pearsonr(pred_flat,labels_flat)[0]
    results["spearman"] = spearmanr(pred_flat,labels_flat)[0]
    results["MAE"] = mean_absolute_error(pred_flat,labels_flat)

    return results

def recall_at_precision():
    ...