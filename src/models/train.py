import argparse
from gc import callbacks
import logging
from operator import mul
from statistics import mode
from typing import Optional

import warnings
import os

from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

from torch.utils.data import DataLoader
from argparse import ArgumentParser

from petastorm import make_reader
from petastorm.pytorch import DataLoader as PetastormDataLoader
logging.getLogger("petastorm").setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

from dotenv import dotenv_values

from src.utils import argparse_to_groups
from src.models.tasks import get_task_with_name
from src.models.models import CNNToTransformerClassifier
from src.utils import get_logger

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

logger = get_logger(__name__)
CONFIG = dotenv_values(".env")

SUPPORTED_MODELS=[
    CNNToTransformerClassifier
]
NAME_MAP = {x.name:x for x in SUPPORTED_MODELS}

def get_model_from_name(name):
    if not name in NAME_MAP:
        msg = f"{name} not recognized as a model name. \
                Supported models are: {NAME_MAP.keys()}"
               
        raise ValueError(msg)        
    return NAME_MAP[name]

def add_model_args(parser,name):
    model = get_model_from_name(name)
    parser = model.add_model_specific_args(parser)
    return parser

def add_task_args(parser,name):
    task = get_task_with_name(name)
    parser = task.add_task_specific_args(parser)
    return parser
    

def add_general_args(parent_parser):
    """ Adds arguments that aren't part of pl.Trainer, but are useful
        (e.g.) --no_wandb """
    parser = parent_parser.add_argument_group("General")
    parser.add_argument("--no_wandb", default=False, action="store_true",
                        help="Run without wandb logging")                        
    parser.add_argument("--train_path", type=str, default=None,
                        help="path to training dataset")
    parser.add_argument("--val_path", type=str, default=None,
                        help="path to validation dataset")  
    parser.add_argument("--notes", type=str, default=None,
                        help="Notes to be sent to WandB")    
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                        help="path to validation dataset")  
    parser.add_argument(
                '-h','--help',
                action='help', default='==SUPPRESS==',
                help='show this help message and exit')                              
    
    return parent_parser               


def train(model_class,task_class, model_args={}, task_args={}, trainer_args={},
          no_wandb : bool = False,
          train_path : str = None,
          val_path : str = None,
          notes : str = None,
          early_stopping_patience : Optional[str] = None,
          **kwargs):
    
    
    task = task_class(**task_args,
                        train_path=train_path,
                        val_path=val_path)
    
    model = model_class(input_shape = task.data_shape,
                        **model_args)

    callbacks = [LearningRateMonitor(logging_interval='step')]

    if not no_wandb:
        # Creating two wandb runs here?
        import wandb
        local_rank = os.environ.get("LOCAL_RANK",0)
        if local_rank == 0:
            logger = WandbLogger(project=CONFIG["WANDB_PROJECT"],
                              entity=CONFIG["WANDB_USERNAME"],
                              notes=notes,
                              log_model=True, #saves checkpoints to wandb as artifacts, might add overhead 
                              reinit=True,
                              resume = 'allow',
                              allow_val_change=True,
                              settings=wandb.Settings(start_method="fork"),
                              id = model.wandb_id)   #id of run to resume from, None if model is not from checkpoint. Alternative: directly use id = model.logger.experiment.id, or try setting WANDB_RUN_ID env variable                
            logger.experiment.summary["task"] = task.get_name()
            logger.experiment.summary["model"] = model.name
            logger.experiment.config.update(model.hparams, allow_val_change=True)
            model.wandb_id = logger.experiment.id  
            
            
        else:
            logger = True
    
    # Set up checkpoint criteria 
    if val_path:
        if task.is_classification:
            checkpoint_metric = "eval/roc_auc"
            mode = "max"
        else:
            checkpoint_metric = "eval/loss"
            mode = "min"

        if early_stopping_patience:
            early_stopping_callback = EarlyStopping(monitor=checkpoint_metric,
                                                    patience=early_stopping_patience,
                                                    mode=mode)
            callbacks.append(early_stopping_callback)
    else:
        checkpoint_metric = "train/loss"
        mode = "min"

    checkpoint_callback = ModelCheckpoint(
                        filename='{epoch}',
                        save_last=True,
                        save_top_k=3,
                        save_on_train_epoch_end = True,
                        monitor=checkpoint_metric,
                        every_n_epochs=1,
                        mode=mode)
    
    callbacks.append(checkpoint_callback)
    
    trainer_args.update(
        dict(
            checkpoint_callback=True,
            callbacks=callbacks,
            accelerator="ddp",
            terminate_on_nan=True,
            num_sanity_val_steps=0,
            profiler="simple",
            gpus=-1
        )
    )
   
    trainer = Trainer(**trainer_args)

    if task.val_url:
        with PetastormDataLoader(make_reader(task.train_url,transform_spec=task.transform,
                                predicate=task.predicate),
                            batch_size=model.batch_size) as train_dataset:
            with PetastormDataLoader(make_reader(task.val_url,transform_spec=task.transform,
                                    predicate=task.predicate),
                                batch_size=model.batch_size) as eval_dataset:
                trainer.fit(model,train_dataset,eval_dataset)
    else:
        with PetastormDataLoader(make_reader(task.train_url,transform_spec=task.transform),
                                batch_size=model.batch_size) as train_dataset:
                trainer.fit(model, train_dataset, DataLoader([["dummy"]]))
   
    logger.info(f"Best model score: {checkpoint_callback.best_model_score}")
    logger.info(f"Best model path: {checkpoint_callback.best_model_path}")
   

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    # figure out which model to use
    # TODO: Optionally, pass a config. Config arguments could just go right into 
    parser.add_argument("--model_name", type=str, default="CNNToTransformerClassifier", 
                                        help= f"Supported models are: {list(NAME_MAP.keys())}")
                                        
    
    # figure out which task to use
    parser.add_argument("--task_name", type=str, default="PredictFluPos", 
                                        help= f"Supported models are: {NAME_MAP.keys()}")

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()
    model_name = temp_args.model_name
    parser = add_model_args(parser,model_name)

    model_class = NAME_MAP[model_name]

    # THIS PULLS THE TASK NAME
    task_name = temp_args.task_name
    parser = add_task_args(parser,task_name)
    task_class = get_task_with_name(task_name)

    # Adds generic trainer args
    parser = Trainer.add_argparse_args(parser)

    # Have to add this down here:
    parser = add_general_args(parser)

    args = parser.parse_args()
    arg_groups = argparse_to_groups(args,parser)

    model_args = vars(arg_groups.get(model_name,argparse.Namespace()))
    task_args = vars(arg_groups.get("Task",argparse.Namespace()))
    trainer_args = vars(arg_groups.get("pl.Trainer",argparse.Namespace()))
    general_args = vars(arg_groups.get("General",argparse.Namespace()))

    train(model_class, task_class,
          model_args=model_args,
          task_args=task_args,
          trainer_args=trainer_args,
          **general_args)