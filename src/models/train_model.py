"""
========================
Model Training Utilities 
========================
`Project repository available here  <https://github.com/behavioral-data/SeattleFluStudy>`_

This module provides the functions to fit different statistical models to the data and evaluating their performance. 
The operations common to all functions regard instantiating the needed model, fetching the train and evaluation data by instatiating
the correct task in the `tasks` module, evaluating the results. 
Some functions also rely on the two utilites `run_huggingface` and `run_pytorch_lightning` in order to fit the model to the data, saving it and
getting evaluation metrics 

**Functions**
    :function train_neural_baseline: 
    :function train_autoencoder:
    :function train_cnn_transformer:
    :function train_sand:
    :function train_bert:
    :function train_longformer:
    :function run_huggingface:
    :function run_pytorch_lightning:
"""
__docformat__ = 'reStructuredText'

import logging
from operator import mul

import warnings
import os
import petastorm
import wandb

logging.getLogger("petastorm").setLevel(logging.ERROR)

from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers.base import DummyExperiment
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

from torch.utils.data import DataLoader

from petastorm import make_reader
from petastorm.pytorch import DataLoader as PetastormDataLoader

import pickle

from torch import tensor
warnings.filterwarnings("ignore")

from dotenv import dotenv_values


from src.models.autoencode import get_autoencoder_by_name, run_autoencoder
from src.models.tasks import get_task_with_name, Autoencode
from src.models.neural_baselines import create_neural_model
from src.models.models import CNNToTransformerClassifier
from src.SAnD.core.model import SAnD
from src.utils import (get_logger, load_dotenv, render_network_plot, set_gpus_automatically, 
                        visualize_model)
from src.data.utils import write_dict_to_json
from src.models.load_model import load_model_from_huggingface_checkpoint



import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import pandas as pd
from PIL import Image
import torch

logger = get_logger(__name__)
CONFIG = dotenv_values(".env")

def train_neural_baseline(model_name,task_name,
                         model_path=None,
                         n_epochs=10,
                         no_early_stopping=False,
                         pos_class_weight = 100,
                         neg_class_weight = 1,
                         eval_frac = 0.15,
                         no_wandb=False,
                         notes=None,
                         dataset_args = {},
                         activity_level="minute",
                         look_for_cached_datareader=False,
                         datareader_ray_obj_ref=None,
                         data_location=None):

    if model_path:
        raise NotImplementedError()

    logger.info(f"Training {model_name} on {task_name}")
    dataset_args["eval_frac"] = eval_frac
    dataset_args["data_location"] = data_location
    task = get_task_with_name(task_name)(dataset_args=dataset_args, activity_level=activity_level, #get task class and input the arguments 
                                        look_for_cached_datareader=look_for_cached_datareader,
                                        datareader_ray_obj_ref=datareader_ray_obj_ref)

    train_X, train_y = task.get_train_dataset().to_stacked_numpy()
    eval_X, eval_y  = task.get_eval_dataset().to_stacked_numpy()

    infer_example = train_X[0]
    n_timesteps, n_features = infer_example.shape
    
    model = create_neural_model(model_name, n_timesteps,n_features)
    config_info = {"n_epochs": n_epochs,
                   "pos_class_weight": pos_class_weight,
                   "neg_class_weight": neg_class_weight,
                   "model_type":model_name,
                   "task":task.get_name(),
                   "dataset_args":dataset_args}
    
    train_class_balance = pd.Series(train_y).value_counts().to_dict()
    logger.info(f"Train class balance: {train_class_balance}")

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
                   entity="mikeamerrill", #TODO make this an argument? could also be a utility function in utils module
                   config=config_info,
                   notes=notes)
        wandb.log({"train_class_balance":train_class_balance})                   
        callbacks.append(WandbCallback())
    else:
        logger.info(f"Config: {config_info}")


    logger.info(f"Training {model_name}")
    model.fit(train_X, train_y, 
            class_weight = {1: pos_class_weight, 0: neg_class_weight}, 
            epochs=n_epochs, validation_split=eval_frac, 
            callbacks = callbacks, verbose=1)
    if len(eval_X) > 0:
        logger.info(f"Training complete. Running evaluation...")
        pred_prob = model.predict(eval_X, verbose=0)
        results = task.evaluate_results(pred_prob,eval_y)
        logger.info("Eval results...")
        logger.info(results)


def train_cnn_transformer( 
                model_config={},
                task_config=None,
                task_name=None, 
                n_epochs=10,
                hidden_size=768,
                num_attention_heads=4,
                num_hidden_layers=4,
                max_length = 24*60+1,
                model_path=None,
                max_position_embeddings=2048, 
                no_early_stopping=False,
                only_with_lab_results=False,
                train_batch_size = 4,
                eval_batch_size = 16,
                eval_frac = None,
                learning_rate = 5e-5,
                classification_threshold=0.5,
                warmup_steps=500,
                weight_decay=0.1,
                no_wandb=False,
                notes=None,
                sinu_position_encoding = False,
                dataset_args = {},
                activity_level="minute",
                look_for_cached_datareader=False,
                cached_task_path = None,
                datareader_ray_obj_ref=None,
                task_ray_obj_ref=None, 
                data_location=None,
                no_eval_during_training=False,
                reset_cls_params=False,
                use_huggingface=False,
                limit_train_frac=None,
                freeze_encoder=False,
                tune=False,
                resume_model_from_ckpt=None,
                output_dir=None,
                kernel_sizes = [5,3,2],
                out_channels = [256,128,64],
                stride_sizes = [3,2,2],
                backend="petastorm",
                train_path=None,
                eval_path=None,
                test_path=None,
                val_epochs=10,
                auto_set_gpu=None,
                dropout_rate=0.5,
                train_mix_positives_back_in=False, 
                log_steps=50,
                train_mixin_batch_size=3,
                pl_seed=2494,
                multitask_daily_features=False,
                downsample_negative_frac=None,
                reload_dataloaders = 0, #to be passed to the pytorch lightning Trainer instance. Reload dataloaders every n epochs (default 0, don't reload)
                early_stopping=False,
                no_bootstrap=False,
                **model_specific_kwargs):

    if auto_set_gpu:
        set_gpus_automatically(auto_set_gpu)
    
    if pl_seed:
        pl.seed_everything(pl_seed)
        wandb.log({"pl_seed": pl_seed})


    logger.info(f"Training CNNTransformer")
    
    if task_config:
        task_name = task_config.get("task_name")
        task_args = task_config.get("task_args",{})
        dataset_args = task_config.get("dataset_args",{})
    else:
        task_name = None
        task_args = None

    if not eval_frac is None:
        dataset_args["eval_frac"] = eval_frac

    dataset_args["limit_train_frac"]=limit_train_frac
    dataset_args["return_dict"] = True
    dataset_args["data_location"] = data_location
    dataset_args["limit_train_frac"] = limit_train_frac
    
    if sinu_position_encoding:
        dataset_args["add_absolute_embedding"] = True

    if cached_task_path:
        logger.info(f"Loading pickle from {cached_task_path}...")
        task = pickle.load(open(cached_task_path,"rb"))



    task = get_task_with_name(task_name)( **task_args,
                                            downsample_negative_frac=downsample_negative_frac,
                                            dataset_args=dataset_args,
                                            activity_level=activity_level,
                                            look_for_cached_datareader=look_for_cached_datareader,
                                            only_with_lab_results = only_with_lab_results,
                                            datareader_ray_obj_ref=datareader_ray_obj_ref,
                                            backend=backend,
                                            append_daily_features=multitask_daily_features,
                                            train_path=train_path,
                                            eval_path=eval_path,
                                            test_path=test_path)
    

    n_timesteps, n_features = task.data_shape  

    if task.is_classification:
        model_head = "classification"
        num_labels = 2

    elif task.is_autoencoder:
        model_head = "autoencoder"
        num_labels = n_timesteps
    else:
        model_head = "regression"
        num_labels = task.labler.label_size
        
    model_kwargs = dict(input_features=n_features,
                            n_timesteps=n_timesteps,
                            num_attention_heads = num_attention_heads,
                            num_hidden_layers = num_hidden_layers,
                            num_labels=num_labels,
                            learning_rate =learning_rate,
                            warmup_steps = warmup_steps,
                            inital_batch_size=train_batch_size,
                            dropout_rate=dropout_rate,
                            kernel_sizes=kernel_sizes,
                            stride_sizes=stride_sizes,
                            out_channels=out_channels,
                            train_mixin_batch_size = train_mixin_batch_size,
                            train_mix_positives_back_in = train_mix_positives_back_in,
                            model_head=model_head,
                            no_bootstrap=no_bootstrap,
                            multitask_daily_features=multitask_daily_features,
                            **model_specific_kwargs)
    if model_config:
        model_kwargs.update(model_config) 
    if model_path:
        if use_huggingface:
            model = load_model_from_huggingface_checkpoint(model_path)
        else:
            model = CNNToTransformerClassifier.load_from_checkpoint(model_path, 
                                                                strict=False,
                                                                **model_kwargs)
            # If using this arg we typically don't want to override a wandb run
            model.hparams.wandb_id = None

    elif resume_model_from_ckpt:
            model = CNNToTransformerClassifier.load_from_checkpoint(resume_model_from_ckpt, 
                                                                strict=False, **model_specific_kwargs)                                                          
    else:
        model = CNNToTransformerClassifier(**model_kwargs)

    if reset_cls_params and hasattr(model,"clf"):
        model.head.reset_parameters()

    if freeze_encoder:
        for param in model.blocks.parameters():
            param.requires_grad = False
        for param in model.input_embedding.parameters():
            param.requires_grad = False


    if output_dir:
        results_dir = os.path.join(output_dir,"results")
        logging_dir = os.path.join(output_dir,"logs")

        os.mkdir(results_dir)
        os.mkdir(logging_dir)
    else:
        results_dir = './results'
        logging_dir = './logs'
    print(results_dir)
    if no_wandb:
        report_to = []
    else:
        report_to = ["wandb"]


    pl_training_args = dict(
        max_epochs=n_epochs,
        check_val_every_n_epoch=val_epochs,
        resume_from_checkpoint=resume_model_from_ckpt,
        log_every_n_steps=log_steps
    )
    
    run_pytorch_lightning(model,task,training_args=pl_training_args,backend=backend, 
                            reload_dataloaders = reload_dataloaders)





def run_pytorch_lightning(model, task, 
                        training_args={},
                        no_wandb=False,
                        notes=None,
                        backend="petastorm",
                        reload_dataloaders = 0,
                        early_stopping=False): #to be passed to the pytorch lightning Trainer instance. Reload dataloaders every n epochs (default 0, don't reload)      

    
    callbacks = [LearningRateMonitor(logging_interval='step')]
    
    if backend == "petastorm":
        do_eval = bool(task.eval_url)
    else:
        do_eval = hasattr(task,"eval_dataset")

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
            #model_img_path = visualize_model(model, dir=wandb.run.dir)
            # wandb.log({"model_img": [wandb.Image(Image.open(model_img_path), caption="Model Graph")]})
            
        else:
            logger = True

        if do_eval:
            if task.is_classification:
                checkpoint_metric = "eval/roc_auc"
                mode = "max"
            else:
                checkpoint_metric = "eval/loss"
                mode = "min"

            if early_stopping:
                early_stopping_callback = EarlyStopping(monitor=checkpoint_metric,patience=2,mode=mode)
                callbacks.append(early_stopping_callback)
        else:
            checkpoint_metric = "train/loss"
            mode = "min"

        checkpoint_callback = ModelCheckpoint(
                            # dirpath=logger.experiment.dir,
                            filename='{epoch}-',
                            # save_last=True,
                            save_top_k=3,
                            save_on_train_epoch_end = True,
                            monitor=checkpoint_metric,
                            every_n_epochs=1,
                            mode=mode)
        
        callbacks.append(checkpoint_callback)

    else:
        checkpoint_callback = True
        logger=True
    
    debug_mode = os.environ.get("DEBUG_MODE")
    trainer = pl.Trainer(logger=logger,
                         checkpoint_callback=True,
                         callbacks=callbacks,
                         gpus = -1,
                         accelerator="ddp",
                         terminate_on_nan=True,
                         num_sanity_val_steps=0,
                         limit_val_batches= 0.0 if not do_eval else 1.0,
                         limit_train_batches=10 if debug_mode else 1.0,
                         profiler="simple",
                         reload_dataloaders_every_n_epochs = reload_dataloaders, #how often to reload dataloaders (defaut:every 0 epochs)
                         **training_args)

    if backend in ["dask", "dynamic"]:
        model.set_train_dataset(task.get_train_dataset())
        model.set_eval_dataset(task.get_eval_dataset())
        trainer.fit(model)
    else:
        ## Manages train and eval context for petastorm:

        if do_eval:
            with PetastormDataLoader(make_reader(task.train_url,transform_spec=task.transform,
                                     predicate=task.predicate),
                                    batch_size=model.batch_size) as train_dataset:
                with PetastormDataLoader(make_reader(task.eval_url,transform_spec=task.transform,
                                     predicate=task.predicate),
                                    batch_size=model.batch_size) as eval_dataset:
                    trainer.fit(model,train_dataset,eval_dataset)
        else:
            with PetastormDataLoader(make_reader(task.train_url,transform_spec=task.transform),
                                    batch_size=model.batch_size) as train_dataset:
                    trainer.fit(model, train_dataset, DataLoader([["dummy"]]))

    print(f"Best model score: {checkpoint_callback.best_model_score}")
    print(f"Best model path: {checkpoint_callback.best_model_path}")

    if task.test_url:
        with PetastormDataLoader(make_reader(task.test_url,transform_spec=task.transform),
                                   batch_size=3*model.batch_size) as test_dataset:
            trainer.test(ckpt_path=checkpoint_callback.best_model_path,
                        test_dataloaders=test_dataset)