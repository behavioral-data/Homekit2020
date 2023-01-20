import os
from typing import Dict, Any, Union, List, Optional, Union, Callable, AnyStr
import numpy as np
from numpy.core.shape_base import vstack
import pandas as pd
from torchmetrics.classification.auroc import AUROC
from torchmetrics.regression import explained_variance
import wandb
import copy
from wandb.plot.roc_curve import roc_curve
from wandb.viz import CustomChart
from wandb.data_types import Table

import torch
import scipy

import torchmetrics
from torchmetrics import (BinnedPrecisionRecallCurve, BinnedAveragePrecision, 
                          BootStrapper, MetricCollection, Metric, CosineSimilarity,
                          ExplainedVariance)
                          
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
        return {str(i):self.counts[i] for i in range(self.n_classes + 1)}

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




class TorchMetricRegression(MetricCollection):
    def __init__(self, bootstrap_samples=1000,
                 prefix=""):
        self.add_prefix = prefix
        metrics = {}

        self.bootstrap_samples = bootstrap_samples

        if bootstrap_samples:
            cosine_sim = BootStrapper(CosineSimilarity(),
                                    num_bootstraps=bootstrap_samples)
            explained_variance = BootStrapper(ExplainedVariance(),
                                  num_bootstraps=bootstrap_samples)
        else:    
            cosine_sim = CosineSimilarity()
            explained_variance = ExplainedVariance()

        metrics["cosine_sim"] = cosine_sim
        metrics["explained_variance"] = explained_variance
        

        self.best_metrics = {"cosine_sim":(max,0),
                             "explained_variance":(max,0)}

        super(TorchMetricRegression,self).__init__(metrics)

    
    def compute(self) -> Dict[str, Any]:
        results = super().compute()
        if self.bootstrap_cis:

            cosine_sim = results["cosine_sim"]["mean"] 
            cosine_sim_std = results["cosine_sim"]["std"] 
            results["cosine_sim_ci_high"] = cosine_sim + 2*cosine_sim_std
            results["cosine_sim_ci_low"] = cosine_sim - 2*cosine_sim_std
            results["cosine_sim"] = results["cosine_sim"]["mean"]
        
            explained_variance = results["explained_variance"]["mean"] 
            explained_variance_std = results["explained_variance"]["std"] 
            results["explained_variance_ci_high"] = explained_variance + 2*explained_variance_std
            results["explained_variance_ci_low"] = explained_variance - 2*explained_variance_std
            results["explained_variance"] = results["explained_variance"]["mean"]
        
        for metric , (operator,old_value) in self.best_metrics.items():
            
            gt_max = (operator == max) and (results[metric] >= old_value)
            lt_min = (operator == min) and (results[metric] <= old_value)
            if gt_max or lt_min:
                self.best_metrics[metric] = (operator,results[metric])
                results[f"best_{metric}"] = results[metric]

        if self.add_prefix:
            return add_prefix(results,self.add_prefix)
        else:
            return results

class TorchMetricClassification(MetricCollection):
    def __init__(self, bootstrap_samples=200,
                 prefix=""):
        
        self.add_prefix = prefix
        self.bootstrap_samples = bootstrap_samples


        metrics = {}

        if bootstrap_samples:
            roc_auc = BootStrapper(torchmetrics.AUROC(), quantile=torch.tensor([0.975, 0.025], device="cuda"),
                                   num_bootstraps=bootstrap_samples)
            pr_auc = BootStrapper(TorchPrecisionRecallAUC(), quantile=torch.tensor([0.975, 0.025], device="cuda"),
                                  num_bootstraps=bootstrap_samples)
        else:    
            roc_auc = torchmetrics.AUROC()  
            pr_auc = TorchPrecisionRecallAUC()

        metrics["roc_auc"] = roc_auc
        metrics["pr_auc"] = pr_auc
        metrics["support"] = Support()

        super(TorchMetricClassification,self).__init__(metrics)

        self.best_metrics = {"pr_auc":(max,0),
                             "roc_auc":(max,0)}

    def compute(self) -> Dict[str, Any]:
        results = {k: m.compute() for k, m in self.items(keep_base=True, copy_state=False)}
        results = {self._set_name(k): v for k, v in results.items()}

        if self.bootstrap_samples:

            # roc_auc = results["roc_auc"]["mean"]
            # roc_std = results["roc_auc"]["std"]
            results["roc_auc_ci_high"] = results["roc_auc"]["quantile"][0]
            results["roc_auc_ci_low"] = results["roc_auc"]["quantile"][1]
            results["roc_auc"] = results["roc_auc"]["mean"]
        
            # pr_auc = results["pr_auc"]["mean"]
            # pr_std = results["pr_auc"]["std"]
            results["pr_auc_ci_high"] = results["pr_auc"]["quantile"][0]
            results["pr_auc_ci_low"] = results["pr_auc"]["quantile"][1]
            results["pr_auc"] = results["pr_auc"]["mean"]
        
        for metric , (operator,old_value) in self.best_metrics.items():
            
            gt_max = (operator == max) and (results[metric] >= old_value)
            lt_min = (operator == min) and (results[metric] <= old_value)
            if gt_max or lt_min:
                self.best_metrics[metric] = (operator,results[metric])
                results[f"best_{metric}"] = results[metric]

        if self.add_prefix:
            return add_prefix(results,self.add_prefix)
        else:
            return results

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore
        probs = torch.nn.functional.softmax(preds,dim=1)[:,-1]

        return super().update(probs, target)

def make_ci_bootsrapper(estimator):
    """  """ 
    def bootstrapper(pred,labels,n_samples=100):
        results = []
        inds = np.arange(len(pred))
        for _ in range(n_samples): #TODO how does it not return an array of same values? JM
            sample = np.random.choice(inds,len(pred)) #sample from the (0,...,len(pred)) vector without replacement for len(pred) times 
            try:
                results.append(estimator(pred[sample],labels[sample])) #get AUC estimates for the shuffled logits vector 
            except ValueError:
                continue
        return np.quantile(results,0.05), np.quantile(results,0.95)
    return bootstrapper

def add_prefix(results,prefix):
    renamed = {}
    for k,v in results.items():
        renamed[prefix+k] = v
    return renamed

def classification_eval(preds, labels, threshold = None, prefix=None,
                        bootstrap_cis=False):
    """
    Get a dictionary containing performance evaluation 
    results for the classification setting. 

    **Arguments** 
        :preds: tensor of unnormalized probabilities 
        :labels: ground truth labels 
        :threshold: 
        :prefix:
        :bootstrap_cis:  
    """

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
    
    probs = torch.nn.functional.softmax(preds,dim=1)[:,-1]
    precision, recall, _thresholds = pr_curve(probs, labels)
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
    
    probs = scipy.special.softmax(preds,axis=1)[:,-1]
    fpr, fnr, _ = det_curve(labels, probs)
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
    probs = torch.nn.functional.softmax(preds,dim=1)[:,-1]
    fpr, tpr, _ = torchmetrics.functional.roc(probs, labels, pos_label=1)
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
    """
    Returns AUC for the classifier
    First computes precision and recall and then returns the torchmetrics.functional.auc() result 
        **Arguments** 
        :pred: tensor of unnormalized probabilities 
    """
    if torch.is_tensor(pred):
        preds = pred.cpu().numpy()
        labels = labels.cpu().numpy()
    else:
        preds = pred
    
    precision, recall, _ = precision_recall_curve(labels,preds)
    result = auc(recall,precision)
    if get_ci:
        ci = make_ci_bootsrapper(pr_auc)(pred,labels,n_samples=n_samples)
        return result, ci
    else:
        return result


def roc_auc(pred,labels,get_ci=False,n_samples=10000):
    """
    Returns AUC for the classifier 
    Uses the sklearn.metrics.roc_auc_score() function
        **Arguments** 
        :pred: tensor of unnormalized probabilities 
    """
    if not isinstance(pred,np.ndarray):
        pred = pred.cpu().numpy()

    score = roc_auc_score(labels,pred)
    if get_ci:
        ci = make_ci_bootsrapper(lambda x,y : roc_auc_score(y_score=x, y_true=y))(pred,labels,n_samples=n_samples)
        return score, ci
    else:
        return score
    
def regression_eval(pred, labels):
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