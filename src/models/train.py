import argparse
from gc import callbacks
import logging
from operator import mul
from statistics import mode
from typing import Optional, Any

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
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

logger = get_logger(__name__)
CONFIG = dotenv_values(".env")


def add_task_args(parser,name):
    task = get_task_with_name(name)
    parser = task.add_task_specific_args(parser)
    return parser
    

def add_general_args(parent_parser):
    """ Adds arguments that aren't part of pl.Trainer, but are useful
        (e.g.) --no_wandb """
    parent_parser.add_argument("--no_wandb", default=False, action="store_true",
                    help="Run without wandb logging")                        
    parent_parser.add_argument("--notes", type=str, default=None,
                    help="Notes to be sent to WandB")    
    parent_parser.add_argument("--early_stopping_patience", type=int, default=None,
                    help="path to validation dataset")                            
    
    return parent_parser               


class CLI(LightningCLI):
    # It's probably possible to use this CLI to train other types of models
    # using custom training loops

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.batch_size", "model.init_args.batch_size",apply_on='parse')
        parser.link_arguments("data.data_shape", "model.init_args.input_shape", apply_on="instantiate")

        add_general_args(parser)
    
    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        # It's probably possible to do all of this from a config.
        # We could set a default config that contains all of this, 
        # which vould be overridden by CLI args. For now,
        # prefer to have the API work like it did in the last project,
        # where we make an educated guess about what we're intended to
        # do based on the model and task that are passed.

        extra_callbacks = []

        if self.datamodule.val_path:
            if self.datamodule.is_classification:
                checkpoint_metric = "eval/roc_auc"
                mode = "max"
            else:
                checkpoint_metric = "eval/loss"
                mode = "min"

            if self.config["fit"]["early_stopping_patience"]:
                early_stopping_callback = EarlyStopping(monitor=checkpoint_metric,
                                                        patience=self.config["fit"]["early_stopping_patience"],
                                                        mode=mode)
                extra_callbacks.append(early_stopping_callback)
        else:
            checkpoint_metric = "train/loss"
            mode = "min"

        self.checkpoint_callback = ModelCheckpoint(
                            filename='{epoch}',
                            save_last=True,
                            save_top_k=3,
                            save_on_train_epoch_end = True,
                            monitor=checkpoint_metric,
                            every_n_epochs=1,
                            mode=mode)
        
        extra_callbacks.append(self.checkpoint_callback)

        local_rank = os.environ.get("LOCAL_RANK",0)
        if not self.config["fit"]["no_wandb"] and local_rank == 0:
            import wandb
            data_logger = WandbLogger(project=CONFIG["WANDB_PROJECT"],
                                entity=CONFIG["WANDB_USERNAME"],
                                notes=self.config["fit"]["notes"],
                                log_model=True, #saves checkpoints to wandb as artifacts, might add overhead 
                                reinit=True,
                                resume = 'allow',
                                allow_val_change=True,
                                settings=wandb.Settings(start_method="fork"),
                                id = self.model.wandb_id)   #id of run to resume from, None if model is not from checkpoint. Alternative: directly use id = model.logger.experiment.id, or try setting WANDB_RUN_ID env variable                
            
            data_logger.experiment.summary["task"] = self.datamodule.get_name()
            data_logger.experiment.summary["model"] = self.model.name
            data_logger.experiment.config.update(self.model.hparams, allow_val_change=True)
            self.model.wandb_id = data_logger.experiment.id  
        
        else:
            data_logger = True   

        extra_callbacks = extra_callbacks + [self._get(self.config_init, c) for c in self._parser(self.subcommand).callback_keys]
        trainer_config = {**self._get(self.config_init, "trainer"), **kwargs}
        return self._instantiate_trainer(trainer_config, extra_callbacks)

    def before_fit(self):
        pass
    
    def after_fit(self):
        logger.info(f"Best model score: {self.checkpoint_callback.best_model_score}")
        logger.info(f"Best model path: {self.checkpoint_callback.best_model_path}")
   
    
    def set_defaults(self):
        ...

if __name__ == "__main__":
    trainer_defaults = dict(
                        checkpoint_callback=True,
                        accelerator="ddp",
                        terminate_on_nan=True,
                        num_sanity_val_steps=0,
                        profiler="simple",
                        gpus=-1,
              )
    
    cli = CLI(trainer_defaults=trainer_defaults)