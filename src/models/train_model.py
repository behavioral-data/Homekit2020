from re import L

import warnings
import os
from pytorch_lightning.loggers.wandb import WandbLogger
from tensorflow.keras import callbacks

from torch import tensor
warnings.filterwarnings("ignore")

import click

from src.models.autoencode import get_autoencoder_by_name, run_autoencoder
from src.models.tasks import get_task_with_name, Autoencode
from src.models.neural_baselines import create_neural_model
from src.models.models import CNNToTransformerEncoder
from src.models.trainer import FluTrainer
from src.SAnD.core.model import SAnD
from src.utils import get_logger, render_network_plot
from src.data.utils import write_dict_to_json
from src.models.load_model import load_model_from_checkpoint

from transformers import (BertForSequenceClassification, Trainer, 
                         TrainingArguments, BertConfig, 
                         EncoderDecoderConfig, EncoderDecoderModel,
                         LongformerForSequenceClassification,
                         LongformerConfig, TransfoXLModel)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import pandas as pd
from PIL import Image
import torch

logger = get_logger(__name__)

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
    task = get_task_with_name(task_name)(dataset_args=dataset_args, activity_level=activity_level,
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
                   entity="mikeamerrill",
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
                no_eval_during_training=False):
    
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


def train_cnn_transformer( task_name, 
                n_epochs=10,
                hidden_size=768,
                num_attention_heads=4,
                num_hidden_layers=4,
                max_length = 24*60+1,
                model_path=None,
                max_position_embeddings=2048, 
                no_early_stopping=False,
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
                reset_cls_params=False,
                use_pl=False,
                limit_train_frac=None,
                freeze_encoder=False,
                **model_specific_kwargs):
    logger.info(f"Training CNNTransformer")
    if not eval_frac is None:
        dataset_args["eval_frac"] = eval_frac
    
    dataset_args["limit_train_frac"]=limit_train_frac
    dataset_args["return_dict"] = True
    dataset_args["data_location"] = data_location
    dataset_args["limit_train_frac"] = limit_train_frac
    
    if sinu_position_encoding:
        dataset_args["add_absolute_embedding"] = True

    task = get_task_with_name(task_name)(dataset_args=dataset_args,
                                        activity_level=activity_level,
                                        look_for_cached_datareader=look_for_cached_datareader,
                                        datareader_ray_obj_ref=datareader_ray_obj_ref)
    
    
    if not model_path:
        train_dataset = task.get_train_dataset()
        infer_example = train_dataset[0]["inputs_embeds"]
        n_timesteps, n_features = infer_example.shape

        model = CNNToTransformerEncoder(input_features=n_features,
                                        n_timesteps=n_timesteps,
                                        num_attention_heads = num_attention_heads,
                                        num_hidden_layers = num_hidden_layers,
                                        num_labels=2,
                                        learning_rate =learning_rate,
                                        warmup_steps = warmup_steps,
                                        inital_batch_size=train_batch_size,
                                        **model_specific_kwargs)
    else:
        model = load_model_from_checkpoint(model_path)
        if reset_cls_params and hasattr(model,"clf"):
            model.clf.reset_parameters()

    if freeze_encoder:
        for param in model.blocks.parameters():
            param.requires_grad = False
        for param in model.input_embedding.parameters():
            param.requires_grad = False
        for param in model.positional_encoding.parameters():
            param.requires_grad = False
            
    if task.is_classification:
        metrics = task.get_huggingface_metrics(threshold=classification_threshold)
    else:
        metrics=None

    if use_pl:
        pl_training_args = dict(
            max_epochs = n_epochs,
            check_val_every_n_epoch=5,
            auto_scale_batch_size="binsearch"
        )
        run_pytorch_lightning(model,task,training_args=pl_training_args)
    
    else:
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
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            prediction_loss_only=False,
            evaluation_strategy="no" if no_eval_during_training else "epoch",
            report_to=["wandb"]            # directory for storing logs
        )

        run_huggingface(model=model, base_trainer=FluTrainer,
                    training_args=training_args,
                    metrics = metrics, task=task,
                    no_wandb=no_wandb, notes=notes)


def train_sand( task_name, 
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
                no_eval_during_training=False):
    
    if model_path:
        raise NotImplementedError()

    logger.info(f"Training SAnD")
    dataset_args["eval_frac"] = eval_frac
    dataset_args["return_dict"] = True
    dataset_args["data_location"] = data_location

    task = get_task_with_name(task_name)(dataset_args=dataset_args, 
                                         activity_level=activity_level,
                                         look_for_cached_datareader=look_for_cached_datareader,
                                         datareader_ray_obj_ref=datareader_ray_obj_ref)
    
    if sinu_position_encoding:
        dataset_args["add_absolute_embedding"] = True


    train_dataset = task.get_train_dataset()
    infer_example = train_dataset[0]["inputs_embeds"]
    n_timesteps, n_features = infer_example.shape

    model = SAnD(input_features=n_timesteps,
                 seq_len = n_features,
                 n_heads = num_attention_heads,
                 factor=256,
                 n_layers = num_hidden_layers,
                 n_class=2,
                 pos_class_weight=pos_class_weight,
                 neg_class_weight=neg_class_weight)
                 

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

    if task.is_classification:
        metrics = task.get_huggingface_metrics(threshold=classification_threshold)
    else:
        metrics=None

    run_huggingface(model=model, base_trainer=FluTrainer,
                   training_args=training_args,
                   metrics = metrics, task=task,
                   no_wandb=no_wandb, notes=notes)

def train_bert(task_name,
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
                data_location=None):
    
    if model_path:
        raise NotImplementedError()

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

def train_longformer(task_name,
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
                    data_location=None):
    
    if model_path:
        raise NotImplementedError()

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
                    metrics, task,no_wandb=False,notes=None):
    if not no_wandb:
        import wandb
        wandb.init(project="flu",
                   entity="mikeamerrill",
                   notes=notes)
        wandb.run.summary["task"] = task.get_name()
        wandb.run.summary["model"] = model.base_model_prefix
        training_args.output_dir = wandb.run.dir
        wandb.log({"run_dir":wandb.run.dir})

    train_dataset = task.get_train_dataset()
    eval_dataset = task.get_eval_dataset()

    if hasattr(model,"config"):
        write_dict_to_json(model.config.to_dict(),
                           os.path.join(training_args.output_dir,"model_config.json"))

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
    train_metrics.pop("_roc")
    train_metrics = {"train/"+k[1:] : v for k,v in train_metrics.items()}
    
    if wandb:
        x_dummy = torch.tensor(train_dataset[0]["inputs_embeds"]).unsqueeze(0).cuda()
        y_dummy = torch.tensor(train_dataset[0]["label"]).unsqueeze(0).cuda()
        pred_dummy = model(inputs_embeds=x_dummy, labels = y_dummy)[0]

        params = dict(model.named_parameters())
        model_img_path = render_network_plot(pred_dummy, wandb.run.dir,params=params )
        wandb.log({"model_img": [wandb.Image(Image.open(model_img_path), caption="Model Graph")]})
        wandb.log(eval_metrics)
        wandb.log(train_metrics)

def run_pytorch_lightning(model, task, 
                        training_args={},
                        no_wandb=False,
                        notes=None):                     
    if not no_wandb:
        logger = WandbLogger(project="flu")
        logger.experiment.notes = notes
        logger.experiment.summary["task"] = task.get_name()
        logger.experiment.summary["model"] = model.base_model_prefix
       
        checkpoint_callback = ModelCheckpoint(
                            dirpath=logger.experiment.dir,
                            filename='{epoch}-',
                            save_last=True)
    else:
        checkpoint_callback = True
        logger=True
    
    if os.environ.get("DEBUG_DATA"):
        training_args["num_sanity_val_steps"] = 0

    trainer = pl.Trainer(logger=logger,
                         checkpoint_callback=checkpoint_callback,
                         gpus = -1,
                         **training_args)

    model.set_train_dataset(task.get_train_dataset())
    model.set_eval_dataset(task.get_eval_dataset())

    trainer.fit(model)
    
