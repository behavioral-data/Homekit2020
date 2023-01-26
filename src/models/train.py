import logging
import os
import warnings
from pprint import pprint
from typing import Optional, Any

import numpy as np
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.trainer.states import TrainerFn

logging.getLogger("petastorm").setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

from dotenv import dotenv_values

from src.models.tasks import get_task_with_name
from src.utils import get_logger
from src.models.loggers import HKWandBLogger as WandbLogger
from src.data.utils import read_parquet_to_pandas
from src.models.eval import classification_eval
from src.utils import upload_pandas_df_to_wandb

from pytorch_lightning.cli import LightningCLI, SaveConfigCallback
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning import Trainer, LightningModule, seed_everything
import wandb
import pandas as pd
from src.models.models.bases import ClassificationModel, NonNeuralMixin
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
    parent_parser.add_argument("--gradient_log_interval", default=0, type=int,
                               help = "Interval with which to log gradients to WandB. 0 -> Never")
    parent_parser.add_argument("--run_name", type=str, default=None,
                               help="run name to use for to WandB")
    parent_parser.add_argument("--pl_seed", type=int, default=2494,
                               help="Pytorch Lightning seed for current experiment")


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

        pl_seed = self.config["fit"]["pl_seed"]
        seed_everything(pl_seed)

        checkpoint_metric = self.config["fit"]["checkpoint_metric"]
        mode = self.config["fit"]["checkpoint_mode"]
        run_name = self.config["fit"]["run_name"]

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
            # filename='{epoch}',
            filename='test_model',
            save_last=True,
            save_top_k=1,
            save_on_train_epoch_end = True,
            monitor=checkpoint_metric,
            every_n_epochs=1,
            mode=mode)

        extra_callbacks.append(self.checkpoint_callback)


        local_rank = os.environ.get("LOCAL_RANK",0)
        if not self.config["fit"]["no_wandb"] and local_rank == 0:
            lr_monitor = LearningRateMonitor(logging_interval='step')
            extra_callbacks.append(lr_monitor)
            # kwargs["save_config_callback"] = WandBSaveConfigCallback

            logger_id = self.model.wandb_id if hasattr(self.model, "id") else None

            data_logger = WandbLogger(project=CONFIG["WANDB_PROJECT"],
                                      entity=CONFIG["WANDB_USERNAME"],
                                      name=run_name,
                                      notes=self.config["fit"]["notes"],
                                      log_model=False, #saves checkpoints to wandb as artifacts, might add overhead
                                      reinit=True,
                                      resume = 'allow',
                                      # save_dir = ".",
                                      allow_val_change=True,
                                      # settings=wandb.Settings(start_method="fork"),
                                      id = logger_id)   #id of run to resume from, None if model is not from checkpoint. Alternative: directly use id = model.logger.experiment.id, or try setting WANDB_RUN_ID env variable

            data_logger.experiment.summary["task"] = os.path.splitext(os.path.basename(str(self.config["fit"]["config"][0])))[0]
            data_logger.experiment.summary["model"] = self.model.name
            data_logger.experiment.summary["pl_seed"] = pl_seed
            data_logger.experiment.summary["checkpoint_metric"] = checkpoint_metric
            data_logger.experiment.config.update(self.model.hparams, allow_val_change=True)
            self.model.wandb_id = data_logger.experiment.id
            self.model.save_hyperparameters()
            # Necessary to save config in the right location
            data_logger._save_dir = data_logger.experiment.dir

        else:
            data_logger = None
        kwargs["logger"] = data_logger

        if isinstance(self.model, NonNeuralMixin):
            return NonNeuralTrainer(self.config["fit"]["no_wandb"])

        extra_callbacks = extra_callbacks + [self._get(self.config_init, c) for c in self._parser(self.subcommand).callback_keys]
        trainer_config = {**self._get(self.config_init, "trainer"), **kwargs}
        return self._instantiate_trainer(trainer_config, extra_callbacks)

    def before_fit(self):
        # Enables logging of gradients to WandB
        gradient_log_interval = self.config["fit"]["gradient_log_interval"]
        if isinstance(self.trainer.logger, WandbLogger) and gradient_log_interval:
            self.trainer.logger.watch(self.model, log="all", log_freq=gradient_log_interval)

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
                if hasattr(self.model, "wandb_id"):
                    self.model.upload_predictions_to_wandb()

        else:
            results = self.trainer.logged_metrics

        if results:
            if self.trainer.logger is not None:
                self.trainer.logger.log_metrics(results)
            else:
                pprint(results)

    def set_defaults(self):
        ...
class NonNeuralTrainer(Trainer):

    def __init__(self, no_wandb=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_wandb = no_wandb

    def fit(self, model, datamodule, *args, **kwargs):
        # ensures the non-neural model has a fit() function
        assert hasattr(model, "fit")

        _, _, x_train, y_train = datamodule.get_train_dataset()
        _, _, x_val, y_val = datamodule.get_val_dataset()

        model.fit(x_train, y_train, eval_set=[(x_val, y_val)]) #, callbacks=[wandb_callback()])

        self.test(model, datamodule, *args, **kwargs)

    def validate(self, model, datamodule, *args, **kwargs):
        assert hasattr(model, "predict")

        _, _, x, y = datamodule.get_val_dataset()

        preds = model.predict_proba(x)[:, 1]
        classification_eval(preds, y, prefix="val/", bootstrap_cis=True)

    def test(self, model, datamodule, *args, **kwargs):
        assert hasattr(model, "predict")

        participant_ids, dates, x, y = datamodule.get_test_dataset()

        preds = model.predict_proba(x)[:, 1]

        results = classification_eval(preds, y, prefix="test/", bootstrap_cis=True)

        if not self.no_wandb:
            wandb.log(results)
            # project = wandb.run.project
            # checkpoint_path = os.path.join(project,wandb.run.id,"checkpoints")
            # os.makedirs(checkpoint_path)

            result_df = pd.DataFrame(zip(participant_ids, dates,  y, preds,),
                                     columns = ["participant_id","date","label","pred"])
            upload_pandas_df_to_wandb(wandb.run.id,"test_predictions",result_df,run=wandb.run)

            # model_path = os.path.join(checkpoint_path, "best.json")
            # model.save_model(model_path)
            # print(f"Saving model to {model_path}")

        print(results)

class NonNeuralTrainer(Trainer):

    def __init__(self, no_wandb=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_wandb = no_wandb

    def fit(self, model, datamodule, *args, **kwargs):
        # ensures the non-neural model has a fit() function
        assert hasattr(model, "fit")

        _, _, x_train, y_train = datamodule.get_train_dataset()
        _, _, x_val, y_val = datamodule.get_val_dataset()

        model.fit(x_train, y_train, eval_set=[(x_val, y_val)])

        self.test(model, datamodule, *args, **kwargs)

    def validate(self, model, datamodule, *args, **kwargs):
        assert hasattr(model, "predict")

        _, _, x, y = datamodule.get_val_dataset()

        preds = model.predict(x)
        classification_eval(preds, y, prefix="val/", bootstrap_cis=True)

    def test(self, model, datamodule, *args, **kwargs):

        assert hasattr(model, "predict")

        participant_ids, dates, x, y = datamodule.get_test_dataset()

        preds = model.predict(x)
        results = classification_eval(preds, y, prefix="test/", bootstrap_cis=True)

        if not self.no_wandb:
            wandb.log(results)
            project = wandb.run.project
            checkpoint_path = os.path.join(project,wandb.run.id,"checkpoints")
            os.makedirs(checkpoint_path)

            result_df = pd.DataFrame(zip(participant_ids, dates,  y, preds,),
                                     columns = ["participant_id","date","label","pred"])

            upload_pandas_df_to_wandb(wandb.run.id,"test_predictions",result_df,run=wandb.run)

        print(results)

if __name__ == "__main__":
    trainer_defaults = dict(
        accelerator="cuda",
        num_sanity_val_steps=0,
        gpus=-1,
    )
    if os.path.isfile("lightning_config.yaml"):
        os.remove("lightning_config.yaml")

    cli = CLI(trainer_defaults=trainer_defaults,
              seed_everything_default=999,
              save_config_overwrite=True)
