from random import choices

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm.notebook import tqdm


def hierarchical_bootstrapping(tables, metrics=None, model_bootstraps=10, data_bootstraps=10, frac=1.0):
    """
    Performs a hierarchical bootstrapping of the given prediction tables
    :param tables: list of prediction tables for all n models
    :param metrics: list of metrics to compute hierarchical bootstrapping over
    :param model_bootstraps: number of times to resample with replacement from the models
    :param model_bootstraps: number of times to resample with replacement from the predictions
    :return: nested dictionary with mean and 95% CI for the bootstrapped metrics
    """

    if metrics is None:
        metrics = [roc_auc_score, average_precision_score]

    final_scores = {m.__name__: [] for m in metrics}

    for _ in tqdm(range(model_bootstraps)):
        scores = {m.__name__: [] for m in metrics}

        # samples with replacement from from models
        for table in choices(tables, k=len(tables)):
            for _ in range(data_bootstraps):

                # samples with replacement from data
                bootstrapped_preds = table.groupby("label", group_keys=False).apply(
                    lambda x: x.sample(frac=frac, replace=True))

                for ix, metric in enumerate(metrics):
                    scores[metric.__name__].append(
                        metric(bootstrapped_preds["label"].to_numpy(int), bootstrapped_preds["pred"].to_numpy()))

        for metric in metrics:
            final_scores[metric.__name__].append(np.mean(scores[metric.__name__]))

    return final_scores
