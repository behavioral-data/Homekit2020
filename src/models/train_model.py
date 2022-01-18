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

import warnings
import os
import petastorm
from ray.util.sgd import data
logging.getLogger("petastorm").setLevel(logging.ERROR)

from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers.base import DummyExperiment
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

from torch.utils.data import DataLoader

from petastorm import make_reader
from petastorm.pytorch import DataLoader as PetastormDataLoader

import ray
from ray.tune.session import report
from tensorflow.keras import callbacks
import pickle

from torch import tensor
warnings.filterwarnings("ignore")

from dotenv import dotenv_values


from src.models.autoencode import get_autoencoder_by_name, run_autoencoder
from src.models.tasks import get_task_with_name, Autoencode
from src.models.neural_baselines import create_neural_model
from src.models.models import CNNToTransformerEncoder
from src.models.trainer import FluTrainer
from src.models.models import LightningSAnD
from src.SAnD.core.model import SAnD
from src.utils import (get_logger, load_dotenv, render_network_plot, set_gpus_automatically, 
                        visualize_model)
from src.data.utils import write_dict_to_json
from src.models.load_model import load_model_from_huggingface_checkpoint

from transformers import (BertForSequenceClassification, Trainer, 
                         TrainingArguments, BertConfig, 
                         EncoderDecoderConfig, EncoderDecoderModel,
                         LongformerForSequenceClassification,
                         LongformerConfig, TransfoXLModel)

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

def train_autoencoder(model_name,
                task_name, 
                n_epochs=10,
                hidden_size=768,
                model_path=None,
                num_attention_heads=4,
                num_hidden_layers=4,
                max_length = 24*60+1,
                max_position_embeddings=2048, 
                no_early_stopping=False,
                pos_class_weight = 100,
                neg_class_weight = 1,
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
                datareader_ray_obj_ref=None,
                data_location=None,
                no_eval_during_training=False,
                auto_set_gpu=None):
    if auto_set_gpu:
        set_gpus_automatically(auto_set_gpu)

    if model_path:
        raise NotImplementedError()

    logger.info(f"Training {model_name}")
    dataset_args["eval_frac"] = eval_frac
    dataset_args["return_dict"] = True
    dataset_args["data_location"] = data_location

    task = get_task_with_name(task_name)(dataset_args=dataset_args, activity_level=activity_level,
                                        look_for_cached_datareader=look_for_cached_datareader,
                                        datareader_ray_obj_ref=datareader_ray_obj_ref)
    
    if sinu_position_encoding:
        dataset_args["add_absolute_embedding"] = True


    train_dataset = task.get_train_dataset()
    infer_example = train_dataset[0]["inputs_embeds"]
    n_timesteps, n_features = infer_example.shape

    base_model = get_autoencoder_by_name(model_name)
    #pylint: disable=unexpected-keyword-arg
    model = base_model(seq_len=n_timesteps, n_features=n_features).cuda()

    training_args = TrainingArguments(
        output_dir='./results',          # output directorz
        num_train_epochs=n_epochs,              # total # of training epochs
        per_device_train_batch_size=train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=eval_batch_size,   # batch size for evaluation
        warmup_steps=warmup_steps,                # number of warmup steps for learning rate scheduler
        weight_decay=weight_decay,
        learning_rate=learning_rate,               # strength of weight decay
        logging_dir='./logs',
        logging_steps=10,
        do_eval=not no_eval_during_training,
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        prediction_loss_only=False,
        evaluation_strategy="epoch",
        report_to=["wandb"]            # directory for storing logs
    )
    metrics = task.get_huggingface_metrics()

    run_huggingface(model=model, base_trainer=FluTrainer,
                   training_args=training_args,
                   metrics = metrics, task=task,
                   no_wandb=no_wandb, notes=notes)


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
                downsample_negative_frac=None,
                reload_dataloaders = 0, #to be passed to the pytorch lightning Trainer instance. Reload dataloaders every n epochs (default 0, don't reload)
                early_stopping=False,
                no_bootstrap=False,
                **model_specific_kwargs):

    if auto_set_gpu:
        set_gpus_automatically(auto_set_gpu)
    
    if pl_seed:
        pl.seed_everything(pl_seed)

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

    elif task_ray_obj_ref:
        task = ray.get(task_ray_obj_ref)
    else:
        task = get_task_with_name(task_name)( **task_args,
                                              downsample_negative_frac=downsample_negative_frac,
                                              dataset_args=dataset_args,
                                              activity_level=activity_level,
                                              look_for_cached_datareader=look_for_cached_datareader,
                                              only_with_lab_results = only_with_lab_results,
                                              datareader_ray_obj_ref=datareader_ray_obj_ref,
                                              backend=backend,
                                              train_path=train_path,
                                              eval_path=eval_path,
                                              test_path=test_path)
    
    if task.is_classification:
        model_head = "classification"
        num_labels = 2
    else:
        model_head = "regression"
        num_labels = task.labler.label_size
    
    n_timesteps, n_features = task.data_shape        
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
                            **model_specific_kwargs)
    if model_config:
        model_kwargs.update(model_config) 
    if model_path:
        if use_huggingface:
            model = load_model_from_huggingface_checkpoint(model_path)
        else:
            model = CNNToTransformerEncoder.load_from_checkpoint(model_path, 
                                                                strict=False,
                                                                **model_kwargs)
            # If using this arg we typically don't want to override a wandb run
            model.hparams.wandb_id = None

    elif resume_model_from_ckpt:
            model = CNNToTransformerEncoder.load_from_checkpoint(resume_model_from_ckpt, 
                                                                strict=False,
                                                                **model_specific_kwargs)                                                          
    else:
        model = CNNToTransformerEncoder(**model_kwargs)

    if reset_cls_params and hasattr(model,"clf"):
        model.head.reset_parameters()

    if freeze_encoder:
        for param in model.blocks.parameters():
            param.requires_grad = False
        for param in model.input_embedding.parameters():
            param.requires_grad = False
        for param in model.positional_encoding.parameters():
            param.requires_grad = False
            
    if tune:
        output_dir = ray.tune.get_trial_dir()

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

    if not use_huggingface:
        pl_training_args = dict(
            max_epochs=n_epochs,
            check_val_every_n_epoch=val_epochs,
            resume_from_checkpoint=resume_model_from_ckpt,
            log_every_n_steps=log_steps
        )
        
        run_pytorch_lightning(model,task,training_args=pl_training_args,backend=backend, 
                                reload_dataloaders = reload_dataloaders,
                                notes=notes, early_stopping=early_stopping)
    else:
        training_args = TrainingArguments(
            output_dir=results_dir,          # output directorz
            num_train_epochs=n_epochs,              # total # of training epochs
            per_device_train_batch_size=train_batch_size,  # batch size per device during training
            per_device_eval_batch_size=eval_batch_size,   # batch size for evaluation
            warmup_steps=warmup_steps,                # number of warmup steps for learning rate scheduler
            weight_decay=weight_decay,
            learning_rate=learning_rate,               # strength of weight decay
            logging_dir=logging_dir,
            logging_steps=10,
            do_eval=not no_eval_during_training,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            prediction_loss_only=False,
            evaluation_strategy="no" if no_eval_during_training else "epoch",
            report_to=report_to            # directory for storing logs
        )

        run_huggingface(model=model, base_trainer=FluTrainer,
                    training_args=training_args,
                    metrics = metrics, task=task,
                    no_wandb=no_wandb, notes=notes,
                    tune=tune)


def train_sand( task_config=None,
                task_name=None,
                n_epochs=10,
                hidden_size=768,
                num_attention_heads=4,
                num_hidden_layers=4,
                max_length = 24*60+1,
                model_path=None,
                max_position_embeddings=2048, 
                no_early_stopping=False,
                pos_class_weight = 1,
                neg_class_weight = 1,
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
                datareader_ray_obj_ref=None,
                data_location=None,
                no_eval_during_training=False,
                auto_set_gpu=None,
                use_huggingface=False,
                backend="petastorm",
                train_path=None,
                eval_path=None,
                test_path=None,
                only_with_lab_results=False,
                downsample_negative_frac=None,
                reload_dataloaders = 0,
                log_steps=50,
                val_epochs=10,
                model_config={},
                dropout_rate=0.5,
                train_mixin_batch_size=3,
                train_mix_positives_back_in=False,
                pl_seed=2494,
                resume_model_from_ckpt=None,
                **model_specific_kwargs):
    
    if auto_set_gpu:
        set_gpus_automatically(auto_set_gpu)
    
    if pl_seed:
        pl.seed_everything(pl_seed)

    logger.info(f"Training SAnD")
    
    if task_config:
        task_name = task_config.get("task_name")                #task_name = task_config["task_name"] 
        dataset_args = task_config.get("dataset_args",{})           #dataset_args = task_config["dataset_args"]
        task_args = task_config.get("task_args",{})                 #task_args = task_config["task_args"]
    else:
        task_name = None
        task_args = None

    if not eval_frac is None:
        dataset_args["eval_frac"] = eval_frac
    dataset_args["return_dict"] = True
    dataset_args["data_location"] = data_location

    
    task = get_task_with_name(task_name)(**task_args,
                                        downsample_negative_frac=downsample_negative_frac,
                                        dataset_args=dataset_args,
                                        activity_level=activity_level,
                                        look_for_cached_datareader=look_for_cached_datareader,
                                        only_with_lab_results = only_with_lab_results,
                                        datareader_ray_obj_ref=datareader_ray_obj_ref,
                                        backend=backend,
                                        train_path=train_path,
                                        eval_path=eval_path,
                                        test_path=test_path)
    
    if sinu_position_encoding:
        dataset_args["add_absolute_embedding"] = True

    if not use_huggingface:
        pl_training_args = dict(
            max_epochs=n_epochs,
            check_val_every_n_epoch=val_epochs,
            log_every_n_steps=log_steps
        )

        """
        train_dataset = task.get_train_dataset()    #better to use task.data_shape
        infer_example = train_dataset[0]["inputs_embeds"]
        n_timesteps, n_features = infer_example.shape
        """
        n_timesteps, n_features = task.data_shape
        model_kwargs = dict(input_features=n_features,
                            n_timesteps=n_timesteps,
                            num_attention_heads = num_attention_heads,
                            num_hidden_layers = num_hidden_layers,
                            num_labels=2,
                            learning_rate =learning_rate,
                            warmup_steps = warmup_steps,
                            inital_batch_size=train_batch_size,
                            dropout_rate=dropout_rate,
                            train_mixin_batch_size = train_mixin_batch_size,
                            train_mix_positives_back_in = train_mix_positives_back_in,
                            d_model=128, #dimensionality of output sequence 
                            factor=256, # Encoder output dimension
                            **model_specific_kwargs)
        if model_config:
        #    model_kwargs.update(model_config)
            raise NotImplementedError
        
        if model_path:
            model = LightningSAnD.load_from_checkpoint(model_path, 
                                                                strict=False,
                                                               **model_specific_kwargs)
            model.hparams.wandb_id = None
            
        elif resume_model_from_ckpt:
            model = LightningSAnD.load_from_checkpoint(resume_model_from_ckpt, 
                                                                strict=False,
                                                                **model_specific_kwargs) 
        else:
            model = LightningSAnD(**model_kwargs)

        run_pytorch_lightning(model = model, 
                        task = task, 
                        training_args=pl_training_args,
                        no_wandb=False,
                        notes=notes,
                        backend=backend,
                        reload_dataloaders = reload_dataloaders)
    
    else:   
 
        train_dataset = task.get_train_dataset()
        infer_example = train_dataset[0]["inputs_embeds"]
        n_timesteps, n_features = infer_example.shape
        if model_path:
            model = load_model_from_huggingface_checkpoint(model_path)   
        else:
            model = SAnD(input_features=n_timesteps,
                        seq_len = n_features,
                        n_heads = num_attention_heads,
                        factor=256,
                        n_layers = num_hidden_layers,
                        n_class=2,
                        pos_class_weight=pos_class_weight,
                        neg_class_weight=neg_class_weight)

        if task.is_classification:
            metrics = task.get_huggingface_metrics(threshold=classification_threshold)
        else:
            metrics=None

        training_args = TrainingArguments(
                        output_dir='./results',          # output directorz
                        num_train_epochs=n_epochs,              # total # of training epochs
                        per_device_train_batch_size=train_batch_size,  # batch size per device during training
                        per_device_eval_batch_size=eval_batch_size,   # batch size for evaluation
                        warmup_steps=warmup_steps,                # number of warmup steps for learning rate scheduler
                        weight_decay=weight_decay,
                        learning_rate=learning_rate,               # strength of weight decay
                        logging_dir='./logs',
                        logging_steps=10,
                        do_eval=not no_eval_during_training,
                        dataloader_num_workers=16,
                        dataloader_pin_memory=True,
                        prediction_loss_only=False,
                        evaluation_strategy="epoch",
                        report_to=["wandb"]            # directory for storing logs
                        )                  
        
        run_huggingface(model=model, base_trainer=FluTrainer,
                   training_args=training_args,
                   metrics = metrics, task=task,
                   no_wandb=no_wandb, notes=notes)



def train_bert(task_config=None,
                task_name=None,
                n_epochs=10,
                hidden_size=768,
                num_attention_heads=4,
                num_hidden_layers=4,
                max_length = 24*60+1,
                max_position_embeddings=2048, 
                model_path=None,
                no_early_stopping=False,
                pos_class_weight = 100,
                neg_class_weight = 1,
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
                datareader_ray_obj_ref=None,
                no_eval_during_training=False,
                auto_set_gpu=None,
                data_location=None):
    
    if auto_set_gpu:
        set_gpus_automatically(auto_set_gpu)

    if model_path:
        raise NotImplementedError()
    if task_config:
        task_name = task_config["task_name"]
        dataset_args = task_config["dataset_args"]

    logger.info(f"Training BERT on {task_name}")
    dataset_args["return_dict"] = True
    dataset_args["eval_frac"] = eval_frac
    dataset_args["data_location"] = data_location
    
    
    if sinu_position_encoding:
        dataset_args["add_absolute_embedding"] = True
        position_embedding_type = None
    else:
        position_embedding_type="absolute"

    task = get_task_with_name(task_name)(dataset_args=dataset_args,
                                         activity_level=activity_level,
                                         look_for_cached_datareader=look_for_cached_datareader)
    
    train_dataset = task.get_train_dataset()
    infer_example = train_dataset[0]["inputs_embeds"]
    n_timesteps, n_features = infer_example.shape


    training_args = TrainingArguments(
        output_dir='./results',          # output directorz
        num_train_epochs=n_epochs,              # total # of training epochs
        per_device_train_batch_size=train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=eval_batch_size,   # batch size for evaluation
        warmup_steps=warmup_steps,                # number of warmup steps for learning rate scheduler
        weight_decay=weight_decay,
        learning_rate=learning_rate,               # strength of weight decay
        logging_dir='./logs',
        logging_steps=10,
        do_eval=not no_eval_during_training,
        prediction_loss_only=False,
        evaluation_strategy="epoch",
        report_to=["wandb"]            # directory for storing logs
    )

    if task.is_classification:
        config = BertConfig(hidden_size=n_features,
                        num_attention_heads=num_attention_heads,
                        num_hidden_layers=num_hidden_layers,
                        max_position_embeddings=n_timesteps,
                        position_embedding_type=position_embedding_type)
        model = BertForSequenceClassification(config)
        model.cuda()

        metrics = task.get_huggingface_metrics(threshold=classification_threshold)
    
    elif task.is_autoencoder:
        raise NotImplementedError
        config = BertConfig(hidden_size=n_features,
                        num_attention_heads=num_attention_heads,
                        num_hidden_layers=num_hidden_layers,
                        max_position_embeddings=n_timesteps,
                        position_embedding_type=position_embedding_type,
                        output_hidden_states=True)
        config  = EncoderDecoderConfig.from_encoder_decoder_configs(config,config)
        model = EncoderDecoderModel(config=config)
        model.config.decoder.is_decoder = True
        model.config.add_cross_attention = True

    run_huggingface(model=model, base_trainer=FluTrainer,
                   training_args=training_args,
                   metrics = metrics, task=task,
                   no_wandb=no_wandb, notes=notes)

def train_longformer(task_config=None,
                    task_name=None,
                    n_epochs=10,
                    hidden_size=768,
                    num_attention_heads=4,
                    num_hidden_layers=4,
                    max_length = 24*60+1,
                    max_position_embeddings=2048, 
                    no_early_stopping=False,
                    pos_class_weight = 100,
                    neg_class_weight = 1,
                    model_path = None,
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
                    datareader_ray_obj_ref=None,
                    no_eval_during_training=False,
                    data_location=None,
                    auto_set_gpu=None):
    if auto_set_gpu:
        set_gpus_automatically(auto_set_gpu)
        
    if model_path:
        raise NotImplementedError()
    
    if task_config:
        task_name = task_config["task_name"]
        dataset_args = task_config["dataset_args"]
        
    logger.info(f"Training Longformer on {task_name}")
    dataset_args["return_dict"] = True
    dataset_args["eval_frac"] = eval_frac
    dataset_args["return_global_attention_mask"] = True
    dataset_args["data_location"] = data_location

    task = get_task_with_name(task_name)(dataset_args=dataset_args,
                                         activity_level=activity_level,
                                         look_for_cached_datareader=look_for_cached_datareader,
                                         datareader_ray_obj_ref=datareader_ray_obj_ref)
    
    train_dataset = task.get_train_dataset()
    infer_example = train_dataset[0]["inputs_embeds"]
    n_timesteps, n_features = infer_example.shape

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=n_epochs,              # total # of training epochs
        per_device_train_batch_size=train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=eval_batch_size,   # batch size for evaluation
        warmup_steps=warmup_steps,                # number of warmup steps for learning rate scheduler
        weight_decay=weight_decay,
        learning_rate=learning_rate,               # strength of weight decay
        logging_dir='./logs',
        logging_steps=10,
        do_eval=not no_eval_during_training,
        prediction_loss_only=False,
        evaluation_strategy="epoch",
        report_to=["wandb"]            # directory for storing logs
    )

    if task.is_classification:
        
        config = LongformerConfig(hidden_size=n_features,
                        num_attention_heads=num_attention_heads,
                        num_hidden_layers=num_hidden_layers,
                        max_position_embeddings=int(n_timesteps*1.25))
        model = LongformerForSequenceClassification(config)
        model.resize_token_embeddings(n_features)
        model.cuda()
        metrics = task.get_huggingface_metrics(threshold=classification_threshold)
    
    else:
        raise NotImplementedError

    run_huggingface(model=model, base_trainer=FluTrainer,
                   training_args=training_args,
                   metrics = metrics, task=task,
                   no_wandb=no_wandb, notes=notes)


def run_huggingface(model,base_trainer,training_args,
                    metrics, task,no_wandb=False,notes=None,
                    tune=True):
                    
    if not no_wandb: 
        import wandb
        wandb.init(project=CONFIG["WANDB_PROJECT"],
                   entity=CONFIG["WANDB_USERNAME"],
                   notes=notes)
        wandb.run.summary["task"] = task.get_name()
        wandb.run.summary["model"] = model.base_model_prefix
        training_args.output_dir = wandb.run.dir
        wandb.log({"run_dir":wandb.run.dir})

    train_dataset = task.get_train_dataset()
    eval_dataset = task.get_eval_dataset()

    if hasattr(model,"config"):
        write_dict_to_json(model.config.to_dict(),
                           os.path.join(training_args.output_dir,"model_config.json"),
                           safe=True)

    training_args.save_total_limit=3
    trainer_args = dict(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=metrics,
            save_eval=True,)

    trainer = base_trainer(**trainer_args)  
    trainer.train()
    # trainer.evaluate()
    eval_metrics = trainer.predict(eval_dataset, metric_key_prefix="").metrics
    eval_metrics = {"eval/"+k[1:] : v for k,v in eval_metrics.items()}

    train_metrics = trainer.predict(train_dataset, metric_key_prefix="",
                                    description="Train").metrics
    train_metrics.pop("_roc",None)
    train_metrics = {"train/"+k[1:] : v for k,v in train_metrics.items()}
    if tune:
        all_metrics = {**eval_metrics,**train_metrics}
        ray.tune.report(**all_metrics)

    if not no_wandb:
        x_dummy = torch.tensor(train_dataset[0]["inputs_embeds"]).unsqueeze(0).cuda()
        y_dummy = torch.tensor(train_dataset[0]["label"]).unsqueeze(0).cuda()
        pred_dummy = model(inputs_embeds=x_dummy, labels = y_dummy)[0]

        params = dict(model.named_parameters())
        model_img_path = render_network_plot(pred_dummy, wandb.run.dir,params=params )
        wandb.log({"model_img": [wandb.Image(Image.open(model_img_path), caption="Model Graph")]})
        wandb.log(eval_metrics)
        wandb.log(train_metrics)
    
    final_dir = os.path.join(training_args.output_dir,"final_model")
    os.makedirs(final_dir)
    print(f"Saving final model to...{final_dir}")
    trainer.save_model(final_dir)


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
                              id = model.hparams.wandb_id)   #id of run to resume from, None if model is not from checkpoint. Alternative: directly use id = model.logger.experiment.id, or try setting WANDB_RUN_ID env variable                
            logger.experiment.summary["task"] = task.get_name()
            logger.experiment.summary["model"] = model.base_model_prefix
            logger.experiment.config.update(model.hparams, allow_val_change=True)
            model.hparams.wandb_id = logger.experiment.id  
            model_img_path = visualize_model(model, dir=wandb.run.dir)
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
                early_stopping_callback = EarlyStopping(monitor=checkpoint_metric,patience=2,mode="max")
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