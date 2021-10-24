import os
from typing import Dict, Any, Union, List, Optional, Union, Callable, AnyStr
import numpy as np
from numpy.core.shape_base import vstack
import pandas as pd
from torchmetrics.classification.auroc import AUROC
import wandb
import copy
from wandb.plot.roc_curve import roc_curve
from wandb.viz import CustomChart
from wandb.data_types import Table

import torch

import torchmetrics
from torchmetrics import BinnedPrecisionRecallCurve, BinnedAveragePrecision, BootStrapper, MetricCollection, Metric
from torchmetrics.functional import auc as tm_auc
from torchmetrics.functional import precision_recall_curve as tm_precision_recall_curve
from torchmetrics.utilities.data import dim_zero_cat

import pytorch_lightning as pl

from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr

from sklearn.metrics import (accuracy_score,precision_recall_fscore_support, roc_auc_score,
                            mean_absolute_error, det_curve, precision_recall_curve, auc)


from functools import partial
from src.utils import check_for_wandb_run

class Support(Metric):
    def __init__(self,  n_classes: int = 1,
                        compute_on_step: bool = True,
                        dist_sync_on_step: bool = False,
                        process_group: Optional[Any] = None,
                        dist_sync_fn: Callable = None) -> None:

        super().__init__(compute_on_step=compute_on_step,
                        dist_sync_on_step=dist_sync_on_step,
                        process_group=process_group,
                        dist_sync_fn=dist_sync_fn)

        self.n_classes = n_classes
        self.add_state("counts", default = torch.zeros(self.n_classes + 1),
                                dist_reduce_fx="sum")

    def update(self, _preds: torch.Tensor, target: torch.Tensor) -> None:
        values = torch.bincount(target, minlength=self.n_classes+1)
        self.counts += values

    def compute(self) -> Dict[AnyStr,torch.Tensor]:
        return {i:self.counts[i] for i in range(self.n_classes + 1)}

class TorchPrecisionRecallAUC(AUROC):
    """A note about this implementation. It would be much more memory
       efficent to use BinnedPrecisionRecallCurve from torchmetrics, 
       but the update step is much slower. This implementation trades
       off memory (it stores every prediction and label) for speed."""
    
    def compute(self) -> torch.Tensor: 
        if not self.mode:
            raise RuntimeError("You have to have determined mode.")

        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        precisions, recalls, _ = tm_precision_recall_curve(preds,target)
        return tm_auc(recalls,precisions)

class TorchMetricClassification(MetricCollection):
    def __init__(self, bootstrap_cis=False,
                 n_boostrap_samples=100,
                 prefix=""):
        
        self.add_prefix = prefix
        self.bootstrap_cis = bootstrap_cis
        metrics = {}

        if bootstrap_cis:
            roc_auc = BootStrapper(torchmetrics.AUROC(),
                                   num_bootstraps=n_boostrap_samples)
            pr_auc = BootStrapper(TorchPrecisionRecallAUC(),
                                  num_bootstraps=n_boostrap_samples)
        else:    
            roc_auc = torchmetrics.AUROC()  
            pr_auc = TorchPrecisionRecallAUC()

        metrics["roc_auc"] = roc_auc
        metrics["pr_auc"] = pr_auc
        metrics["support"] = Support()

        super(TorchMetricClassification,self).__init__(metrics)
    
    def compute(self) -> Dict[str, Any]:
        results = super().compute()
        if self.bootstrap_cis:

            roc_auc = results["roc_auc"]["mean"] 
            roc_std = results["roc_auc"]["std"] 
            results["roc_auc_ci_high"] = roc_auc + 2*roc_std
            results["roc_auc_ci_low"] = roc_auc - 2*roc_std
            results["roc_auc"] = results["roc_auc"]["mean"]
        
            pr_auc = results["pr_auc"]["mean"] 
            pr_std = results["pr_auc"]["std"] 
            results["pr_auc_ci_high"] = pr_auc + 2*pr_std
            results["pr_auc_ci_low"] = pr_auc - 2*pr_std
            results["pr_auc"] = results["pr_auc"]["mean"]
        
        if self.add_prefix:
            return add_prefix(results,self.add_prefix)
        else:
            return results



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

def add_prefix(results,prefix):
    renamed = {}
    for k,v in results.items():
        renamed[prefix+k] = v
    return renamed

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

def wandb_pr_curve(preds,labels,thresholds=50, num_classes=1):
    # preds = preds.cpu().numpy()
    # labels = labels.cpu().numpy()
    # preds = preds.cpu().numpy()
    pr_curve = BinnedPrecisionRecallCurve(thresholds=thresholds, num_classes=num_classes).to(preds.device)
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
    labels = labels.cpu().numpy()
  
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
    labels = labels.cpu().numpy()
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