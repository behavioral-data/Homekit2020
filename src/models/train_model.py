#pylint: disable=import-error
from json import loads
import warnings
warnings.filterwarnings("ignore")

import click
from src.models.tasks import get_task_with_name
from src.models.neural_baselines import create_neural_model
from src.utils import get_logger

from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

logger = get_logger()

@click.command()
@click.argument("model_name")
@click.argument("task_name")
@click.option("--pos_class_weight", default=100)
@click.option("--neg_class_weight", default=1)
@click.option("--val_split", default=0.15)
@click.option("--no_early_stopping",is_flag=True)
@click.option("--no_wandb",is_flag=True)
@click.option('--dataset_args', default={})
def train_neural_baseline(model_name,task_name,
                         n_epochs=10,
                         no_early_stopping=False,
                         pos_class_weight = 100,
                         neg_class_weight = 1,
                         val_split = 0.15,
                         no_wandb=False,
                         dataset_args = {}):

    # Annoyingly need to load all of this into RAM:

    logger.info(f"Training {model_name} on {task_name}")
    dataset_args = loads(dataset_args)
    task = get_task_with_name(task_name)(dataset_args=dataset_args)

    train_X, train_y = task.get_train_dataset().to_stacked_numpy()
    eval_X, eval_y  = task.get_eval_dataset().to_stacked_numpy()

    infer_example = train_X[0]
    n_timesteps, n_features = infer_example.shape
    
    model = create_neural_model(model_name, n_timesteps,n_features)
    config_info = {"n_epochs": n_epochs,
                   "pos_class_weight": pos_class_weight,
                   "neg_class_weight": neg_class_weight,
                   "model_type":model_name,
                   "task":task.get_name()}
    
    train_class_balance = pd.Series(train_y).value_counts().to_dict()

    callbacks = []
    if not no_early_stopping:
        early_stopping_monitor = EarlyStopping(
                    monitor='val_loss',
                    min_delta=0,
                    patience=10,
                    verbose=0,
                    mode='min',
                    baseline=None,
                    restore_best_weights=True
                )
        callbacks.append(early_stopping_monitor)
    if not no_wandb:
        from wandb.keras import WandbCallback
        import wandb
        wandb.init(project="flu",
                   entity="mikeamerrill",
                   config=config_info)
        wandb.log({"train_class_balance":train_class_balance})                   
        callbacks.append(WandbCallback())
    else:
        logger.info(f"Config: {config_info}")
        logger.info(f"Train class balance: {train_class_balance}")

    logger.info(f"Training {model_name}")
    model.fit(train_X, train_y, 
            class_weight = {1: pos_class_weight, 0: neg_class_weight}, 
            epochs=n_epochs, validation_split=val_split, 
            callbacks = callbacks, verbose=1)
    if len(eval_X) > 0:
        logger.info(f"Training complete. Running evaluation...")
        pred_prob = model.predict(eval_X, verbose=0)
        results = task.evaluate_results(pred_prob,eval_y)
        logger.info("Eval results...")
        logger.info(results)

    