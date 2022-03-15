import click
import numpy as np
from json import loads
import xgboost as xgb
from matplotlib import pyplot as plt
import wandb
import os
import pandas as pd

from src.models.tasks import get_task_with_name
from src.utils import get_logger
from src.models.eval import classification_eval
from src.utils import upload_pandas_df_to_wandb

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
                model_path=None,
                add_features_path=None,
                data_location=None,
                limit_train_frac=None,
                max_depth=2,
                day_window_size=7,
                eta=1,
                objective="binary:logistic",
                train_participant_dates=None,
                eval_participant_dates=None,
                test_participant_dates=None,
                **_):

    """ Baseline for classification tasks that uses daily aggregated features"""
    task_name = task_config["task_name"]

    logger.info(f"Training XGBoost on {task_name}")
    task = get_task_with_name(task_name)(activity_level="day",
                                        limit_train_frac=limit_train_frac,
                                        train_participant_dates=train_participant_dates,
                                        eval_participant_dates=eval_participant_dates,
                                        test_participant_dates=test_participant_dates,
                                        **task_config.get("task_args",{})
                                        )
    if not no_wandb:
            wandb.init(project=CONFIG["WANDB_PROJECT"],
                    entity=CONFIG["WANDB_USERNAME"],
                    notes=notes)
            wandb.log({"task":task.get_name()}) 

    if not task.is_classification:
        raise ValueError(f"{task_name} is not a classification task")

    if test_participant_dates:
        test = task.get_test_dataset().to_dmatrix()
    else:
        test = None
        
    if not model_path:
        train = task.get_train_dataset().to_dmatrix()
        eval = task.get_eval_dataset().to_dmatrix()
       
        
        param = {'max_depth': max_depth, 'eta': eta, 'objective': objective}
        param['nthread'] = 4
        param['eval_metric'] = 'auc'
        evallist = [(eval, 'eval'), (train, 'train')]
        callbacks = []
        

        if add_features_path:
            model_name = "XGBoost-ExpertFeatures"
        else:
            model_name = "XGBoost"
        wandb.log({"model":model_name}) 
        callbacks.append(xgb_wandb_callback())
        bst = xgb.train(param, train, 10, evallist, callbacks=callbacks)
    
    else:
        bst = xgb.Booster()
        bst.load_model(model_path)
    
    if not no_wandb:   
        xgb.plot_importance(bst)
        plt.tight_layout()
        wandb.log({"feature_importance": wandb.Image(plt)})
    
    if test:
        test_pred = bst.predict(test)
        results = classification_eval(test_pred,test.get_label(),prefix="test/",bootstrap_cis=True) 
        test_participant_dates = test.user_dates
        
        if not no_wandb:
            wandb.log(results)
            project = wandb.run.project
            checkpoint_path = os.path.join(project,wandb.run.id,"checkpoints")
            os.makedirs(checkpoint_path)

            participant_ids = [pid for pid, _date in test_participant_dates]
            dates = [date for _pid, date in test_participant_dates]
            result_df = pd.DataFrame(zip(participant_ids, dates,  test.get_label(),test_pred,),
                                    columns = ["participant_id","date","label","pred"])
            
            upload_pandas_df_to_wandb(wandb.run.id,"test_predictions",result_df,run=wandb.run)

            model_path = os.path.join(checkpoint_path, "best.json")
            bst.save_model(model_path)
            print(f"Saving model to {model_path}")
        print(results)