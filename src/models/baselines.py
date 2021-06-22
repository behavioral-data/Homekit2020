import click
import numpy as np
from json import loads
from ray import tune
import xgboost as xgb
from matplotlib import pyplot as plt
from xgboost import callback
import wandb
import ray

from src.models.tasks import get_task_with_name
from src.utils import get_logger
from src.models.eval import classification_eval

logger = get_logger(__name__)

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

def train_xgboost(task_name, dataset_args ={},
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
                **_):

    """ Baseline for classification tasks that uses daily aggregated features"""
    logger.info(f"Training XGBoost on {task_name}")
    dataset_args["add_features_path"] = add_features_path
    dataset_args["data_location"] = data_location
    # dataset_args["data_location"] = data_location

    if task_ray_obj_ref:
        task = ray.get(task_ray_obj_ref)
    else:
        task = get_task_with_name(task_name)(dataset_args=dataset_args,
                                         activity_level="day")
    if not task.is_classification:
        raise ValueError(f"{task_name} is not an classification task")

    train = task.get_train_dataset().to_dmatrix()
    if limit_train_frac:
        inds = list(range(int(len(task.get_train_dataset())*limit_train_frac)))
        train = train.slice(inds)
    eval = task.get_eval_dataset().to_dmatrix()
    
    param = {'max_depth': max_depth, 'eta': eta, 'objective': objective}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    evallist = [(eval, 'eval'), (train, 'train')]
    callbacks = []
    if not no_wandb:
        wandb.init(project="flu",
                   entity="mikeamerrill",
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
    results = classification_eval(eval_logits,eval.get_label())


    if not no_wandb:
        xgb.plot_importance(bst)
        plt.tight_layout()
        wandb.log({"feature_importance": wandb.Image(plt)})
        wandb.log(results)

    if tune:
        ray.tune.report(**results)