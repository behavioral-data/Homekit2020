import click
import numpy as np
from json import loads
import xgboost as xgb

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


@click.command(cls=BaseCommand)
@click.argument("task_name")
def train_xgboost(task_name, dataset_args ={},
                no_wandb=False,
                notes=None,
                activity_level="day"):
    """ Baseline for classification tasks that uses daily aggregated features"""

    
    logger.info(f"Training XGBoost on {task_name}")
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
    bst = xgb.train(param, train, 10, evallist)
    