import click
import numpy as np
from json import loads
import xgboost as xgb
from matplotlib import pyplot as plt
from xgboost import callback
import wandb

from src.models.tasks import get_task_with_name
from src.utils import get_logger
from src.models.commands import BaseCommand
logger = get_logger()

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

@click.command(cls=BaseCommand)
@click.argument("task_name")
@click.option("--add_features_path", type = click.Path(dir_okay=False), default=None)
def train_xgboost(task_name, dataset_args ={},
                no_wandb=False,
                notes=None,
                activity_level="day",
                look_for_cached_datareader = False,
                add_features_path=None,
                data_location=None):

    """ Baseline for classification tasks that uses daily aggregated features"""
    logger.info(f"Training XGBoost on {task_name}")
    dataset_args["add_features_path"] = add_features_path
    # dataset_args["data_location"] = data_location

    task = get_task_with_name(task_name)(dataset_args=dataset_args,
                                         activity_level="day")
    if not task.is_classification:
        raise ValueError(f"{task_name} is not an classification task")

    train = task.get_train_dataset().to_dmatrix()
    eval = task.get_eval_dataset().to_dmatrix()
    
    param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    evallist = [(eval, 'eval'), (train, 'train')]
    
    callbacks = []
    if not no_wandb:
        wandb.init(project="flu",
                   entity="mikeamerrill",
                   notes=notes)
        wandb.run.summary["task"] = task.get_name()
        wandb.run.summary["model"] = "XGBoost"
        callbacks.append(xgb_wandb_callback())
    
    bst = xgb.train(param, train, 10, evallist, callbacks=callbacks)
    
    if not no_wandb:
        xgb.plot_importance(bst)
        plt.tight_layout()
        wandb.log({"feature_importance": wandb.Image(plt)})

    