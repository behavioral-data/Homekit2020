import click
import numpy as np
from json import loads
from ray import tune
import xgboost as xgb
from matplotlib import pyplot as plt
from xgboost import callback
import wandb
import ray
import pickle

from src.models.tasks import get_task_with_name
from src.utils import get_logger
from src.models.eval import classification_eval
from dotenv import dotenv_values

logger = get_logger(__name__)
CONFIG = dotenv_values(".env")

@click.command()
@click.argument("task_name")
@click.option('--dataset_args', default={})
def select_random(task_name, dataset_args ={}):
    """ Baseline for autoencoder tasks that takes
        a random example from the train set 
        and treats it as a reconstruction"""

    
    logger.info(f"Using a random example as a baseline on {task_name}")
    dataset_args = loads(dataset_args)
    task = get_task_with_name(task_name)(dataset_args=dataset_args)
    if not task.is_autoencoder:
        raise ValueError(f"{task_name} is not an Autoencode task")

    train_X = task.get_train_dataset().to_stacked_numpy()
    eval_X = task.get_eval_dataset().to_stacked_numpy()

    n_eval = len(eval_X)
    pred_inds = np.random.randint(0,len(train_X),n_eval)
    preds = train_X[pred_inds]

    results = task.evaluate_results(preds,eval_X)
    logger.info(results)

def xgb_wandb_callback():
    def callback(env):
        for k, v in env.evaluation_result_list:
            stage, metric_name = k.split("-")
            if "auc" in metric_name:
                metric_name = "roc_auc"
            wandb.log({f"{stage}/{metric_name}": v}, commit=False)
        wandb.log({})

    return callback

def train_xgboost(task_config,
                no_wandb=False,
                notes=None,
                activity_level="day",
                look_for_cached_datareader = False,
                add_features_path=None,
                data_location=None,
                limit_train_frac=None,
                max_depth=2,
                eta=1,
                objective="binary:logistic",
                task_ray_obj_ref=None,
                only_with_lab_results=False,
                cached_task_path=None,
                **_):

    """ Baseline for classification tasks that uses daily aggregated features"""
    task_name = task_config["task_name"]
    dataset_args = task_config["dataset_args"]

    logger.info(f"Training XGBoost on {task_name}")
    dataset_args["add_features_path"] = add_features_path
    dataset_args["data_location"] = data_location
    dataset_args["data_location"] = data_location
    if limit_train_frac:
        dataset_args["limit_train_frac"] = limit_train_frac

    if task_ray_obj_ref:
        task = ray.get(task_ray_obj_ref)
    elif cached_task_path:
        logger.info(f"Loading pickle from {cached_task_path}...")
        task = pickle.load(open(cached_task_path,"rb"))
    else:
        task = get_task_with_name(task_name)(dataset_args=dataset_args,
                                            only_with_lab_results=only_with_lab_results,
                                            activity_level="day",
                                            backend="dask",
                                            limit_train_frac=limit_train_frac)

    if not task.is_classification:
        raise ValueError(f"{task_name} is not a classification task")

    train = task.get_train_dataset().to_dmatrix()
    eval = task.get_eval_dataset().to_dmatrix()
    
    param = {'max_depth': max_depth, 'eta': eta, 'objective': objective}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    evallist = [(eval, 'eval'), (train, 'train')]
    callbacks = []
    if not no_wandb:
        wandb.init(project=CONFIG["WANDB_PROJECT"],
                   entity=CONFIG["WANDB_USERNAME"],
                   notes=notes)
        wandb.run.summary["task"] = task.get_name()

        if add_features_path:
            model_name = "XGBoost-ExpertFeatures"
        else:
            model_name = "XGBoost"
        wandb.run.summary["model"] = model_name
        callbacks.append(xgb_wandb_callback())
    
    bst = xgb.train(param, train, 10, evallist, callbacks=callbacks)
    eval_pred = bst.predict(eval)
    eval_logits = np.stack([1-eval_pred,eval_pred],axis=1)
    results = classification_eval(eval_pred,eval.get_label(),prefix="eval/") # results = classification_eval(eval_logits,eval.get_label(),prefix="eval/") JM


    if not no_wandb:
        xgb.plot_importance(bst)
        plt.tight_layout()
        wandb.log({"feature_importance": wandb.Image(plt)})
        wandb.log(results)

    if tune:
        ray.tune.report(**results)