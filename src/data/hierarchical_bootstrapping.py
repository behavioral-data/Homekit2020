from random import choices

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm.notebook import tqdm


def hierarchical_bootstrapping(tables, metrics=None, num_bootstraps=10):
    """
    Performs a hierarchical bootstrapping of the given prediction tables
    :param tables: list of prediction tables for all n models
    :param metrics: list of metrics to compute hierarchical bootstrapping over
    :param num_bootstraps: number of times to resample with replacement from the models
    :return: nested dictionary with mean and 95% CI for the bootstrapped metrics
    """

    if metrics is None:
        metrics = [roc_auc_score, average_precision_score]

    final_scores = {m.__name__: [] for m in metrics}

    for _ in tqdm(range(num_bootstraps)):
        scores = {m.__name__: [] for m in metrics}

        # samples with replacement from from models
        resampled_tables = choices(tables, k=len(tables))

        len_data = len(resampled_tables[0])

        # samples with replacement from data
        resampled_data_indices = np.random.choice(np.arange(len_data), len_data, replace=True)

        for table in resampled_tables:
            bootstrapped_preds = table.iloc[resampled_data_indices]
            labels = bootstrapped_preds["label"].to_numpy(int)
            preds = bootstrapped_preds["pred"].to_numpy()
            for ix, metric in enumerate(metrics):
                scores[metric.__name__].append(metric(labels, preds))

        for metric in metrics:
            final_scores[metric.__name__].append(np.mean(scores[metric.__name__]))

    return final_scores
