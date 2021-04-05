import click
import numpy as np
from json import loads

from src.models.tasks import get_task_with_name
from src.utils import get_logger
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

    train_X = task.get_eval_dataset().to_stacked_numpy()
    eval_X = task.get_eval_dataset().to_stacked_numpy()

    n_eval = len(eval_X)
    pred_inds = np.random.randint(0,len(train_X),n_eval)
    preds = train_X[pred_inds]

    results = task.evaluate_results(preds,eval_X)
    logger.info(results)