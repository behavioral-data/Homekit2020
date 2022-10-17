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
from pytorch_lightning.trainer.states import TrainerFn

from petastorm.pytorch import DataLoader as PetastormDataLoader
logging.getLogger("petastorm").setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

from dotenv import dotenv_values

from src.models.tasks import get_task_with_name
from src.utils import get_logger

from pytorch_lightning.utilities.cli import LightningCLI, SaveConfigCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning import Trainer, LightningModule 

logger = get_logger(__name__)
CONFIG = dotenv_values(".env")


def add_task_args(parser,name):
    task = get_task_with_name(name)
    parser = task.add_task_specific_args(parser)
    return parser
    

def add_general_args(parent_parser):
    """ Adds arguments that aren't part of pl.Trainer, but are useful
        (e.g.) --no_wandb """
    parent_parser.add_argument("--checkpoint_metric", type=str, default=None,
                    help="Metric to optimize for during training")
    parent_parser.add_argument("--checkpoint_mode", type=str, default="max",
                    help="Metric direction to optimize for during training")                    
    parent_parser.add_argument("--no_wandb", default=False, action="store_true",
                    help="Run without wandb logging")                        
    parent_parser.add_argument("--notes", type=str, default=None,
                    help="Notes to be sent to WandB")    
    parent_parser.add_argument("--early_stopping_patience", type=int, default=None,
                    help="path to validation dataset")                            
    
    return parent_parser               

class WandBSaveConfigCallback(SaveConfigCallback):
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:

        if isinstance(trainer.logger, WandbLogger):
            # If we're at rank zero and using WandBLogger then we probably want to
            # log the config
            log_dir = trainer.logger.experiment.dir
            fs = get_filesystem(log_dir)

            config_path = os.path.join(log_dir, self.config_filename)
            fs.makedirs(log_dir, exist_ok=True)
            self.parser.save(
                self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
            )
        else:
            super().setup(trainer,pl_module,stage=stage)

class CLI(LightningCLI):
    # It's probably possible to use this CLI to train other types of models
    # using custom training loops

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.init_args.batch_size","data.init_args.batch_size",apply_on="parse")
        parser.link_arguments("data.data_shape", "model.init_args.input_shape", apply_on="instantiate")
        # parser.link_arguments("trainer.fit_loop", "model.fit_loop", apply_on="instantiate")

        add_general_args(parser)
    
    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        # It's probably possible to do all of this from a config.
        # We could set a default config that contains all of this, 
        # which vould be overridden by CLI args. For now,
        # prefer to have the API work like it did in the last project,
        # where we make an educated guess about what we're intended to
        # do based on the model and task that are passed.

        extra_callbacks = []
        
        checkpoint_metric = self.config["fit"]["checkpoint_metric"]
        mode = self.config["fit"]["checkpoint_mode"]
            
        if "loss" in checkpoint_metric and mode == "max":
                 logger.warning("Maximizing {}".format(checkpoint_metric))
            
        if self.datamodule.val_path:
           
            if self.datamodule.is_classification:
                if checkpoint_metric is None:            
                    checkpoint_metric = "val/roc_auc"
                    mode = "max"
            else:
                 if checkpoint_metric is None:            
                    checkpoint_metric = "val/loss"
                    mode = "min"
            
            if self.config["fit"]["early_stopping_patience"]:
                early_stopping_callback = EarlyStopping(monitor=checkpoint_metric,
                                                        patience=self.config["fit"]["early_stopping_patience"],
                                                        mode=mode)
                extra_callbacks.append(early_stopping_callback)
        else:
            if checkpoint_metric is None:            
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
            # kwargs["save_config_callback"] = WandBSaveConfigCallback
            data_logger = WandbLogger(project=CONFIG["WANDB_PROJECT"],
                                entity=CONFIG["WANDB_USERNAME"],
                                notes=self.config["fit"]["notes"],
                                log_model=True, #saves checkpoints to wandb as artifacts, might add overhead 
                                reinit=True,
                                resume = 'allow',
                                # save_dir = ".",
                                allow_val_change=True,
                                # settings=wandb.Settings(start_method="fork"),
                                id = self.model.wandb_id)   #id of run to resume from, None if model is not from checkpoint. Alternative: directly use id = model.logger.experiment.id, or try setting WANDB_RUN_ID env variable                
            
            data_logger.experiment.summary["task"] = self.datamodule.get_name()
            data_logger.experiment.summary["model"] = self.model.name
            data_logger.experiment.config.update(self.model.hparams, allow_val_change=True)
            self.model.wandb_id = data_logger.experiment.id  
            
            # Necessary to save config in the right location
            data_logger._save_dir = data_logger.experiment.dir
        
        else:
            data_logger = None  
        kwargs["logger"] = data_logger
        
        extra_callbacks = extra_callbacks + [self._get(self.config_init, c) for c in self._parser(self.subcommand).callback_keys]
        trainer_config = {**self._get(self.config_init, "trainer"), **kwargs}
        return self._instantiate_trainer(trainer_config, extra_callbacks)

    def after_fit(self):
        if self.trainer.is_global_zero:
            logger.info(f"Best model score: {self.checkpoint_callback.best_model_score}")
            logger.info(f"Best model path: {self.checkpoint_callback.best_model_path}")
        results = {}
        
        if self.trainer.state.fn == TrainerFn.FITTING:
            if (
                self.trainer.checkpoint_callback
                and self.trainer.checkpoint_callback.best_model_path
            ):
                ckpt_path = self.trainer.checkpoint_callback.best_model_path
                # Disable useless logging
                logging.getLogger("pytorch_lightning.utilities.distributed").setLevel(
                    logging.WARNING
                )
                logging.getLogger("pytorch_lightning.accelerators.gpu").setLevel(
                    logging.WARNING
                )

                self.trainer.callbacks = []
                fn_kwargs = {
                    "model": self.model,
                    "datamodule": self.datamodule,
                    "ckpt_path": ckpt_path,
                    "verbose": False,
                }

                has_test_loader = (
                    self.trainer._data_connector._test_dataloader_source.is_defined()
                )

                results = self.trainer.test(**fn_kwargs)[0] if has_test_loader else {}

        else:
            results = self.trainer.logged_metrics

        if results:
            self.trainer.logger.log_metrics(results)

    def set_defaults(self):
        ...

if __name__ == "__main__":
    trainer_defaults = dict(
                        accelerator="cuda",
                        num_sanity_val_steps=0,
                        gpus=-1,
              )
    if os.path.isfile("lightning_config.yaml"):
        os.remove("lightning_config.yaml")
    
    cli = CLI(trainer_defaults=trainer_defaults,
            #  save_config_callback=WandBSaveConfigCallback,
            seed_everything_default=999,
            save_config_filename="lightning_config.yaml")
