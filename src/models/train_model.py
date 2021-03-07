#pylint: disable=import-error
from json import loads
import warnings
warnings.filterwarnings("ignore")

import click
from src.models.tasks import get_task_with_name
from src.models.neural_baselines import create_neural_model
from src.utils import get_logger

from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertConfig

from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

logger = get_logger()

@click.command()
@click.argument("model_name")
@click.argument("task_name")
@click.option("--n_epochs", default=10)
@click.option("--pos_class_weight", default=100)
@click.option("--neg_class_weight", default=1)
@click.option("--val_split", default=0.15)
@click.option("--no_early_stopping",is_flag=True)
@click.option("--no_wandb",is_flag=True)
@click.option("--notes", type=str, default=None, help="Notes to save to wandb")
@click.option('--dataset_args', default={})
def train_neural_baseline(model_name,task_name,
                         n_epochs=10,
                         no_early_stopping=False,
                         pos_class_weight = 100,
                         neg_class_weight = 1,
                         val_split = 0.15,
                         no_wandb=False,
                         notes=None,
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
            epochs=n_epochs, validation_split=val_split, 
            callbacks = callbacks, verbose=1)
    if len(eval_X) > 0:
        logger.info(f"Training complete. Running evaluation...")
        pred_prob = model.predict(eval_X, verbose=0)
        results = task.evaluate_results(pred_prob,eval_y)
        logger.info("Eval results...")
        logger.info(results)


@click.command()
@click.argument("task_name")
# Model Args:
@click.option("--n_epochs", default=10)
@click.option("--hidden_size",default=768)
@click.option("--num_attention_heads", default=4)
@click.option("--num_hidden_layers", default=4)
@click.option("--max_length", default=24*60+1)
@click.option("--max_position_embeddings",default=2048)
# Training Args:
@click.option("--pos_class_weight", default=100)
@click.option("--neg_class_weight", default=1)
@click.option("--train_batch_size",default=20)
@click.option("--eval_batch_size",default=60)
@click.option("--no_early_stopping",is_flag=True)
@click.option("--warmup_steps",default=500)
@click.option("--weight_decay",default=0.1)
@click.option("--eval_frac",default=0.15)
@click.option("--classification_threshold",default=0.5)
# WandB Args:
@click.option("--no_wandb",is_flag=True)
@click.option("--notes", type=str, default=None, help="Notes to save to wandb")
@click.option('--dataset_args', default={})
def train_bert(task_name,
                n_epochs=10,
                hidden_size=768,
                num_attention_heads=4,
                num_hidden_layers=4,
                max_length = 24*60+1,
                max_position_embeddings=2048, 
                no_early_stopping=False,
                pos_class_weight = 100,
                neg_class_weight = 1,
                train_batch_size = 4,
                eval_batch_size = 16,
                eval_frac = 0.15,
                classification_threshold=0.5,
                warmup_steps=500,
                weight_decay=0.1,
                no_wandb=False,
                notes=None,
                dataset_args = {}):

    logger.info(f"Training BERT on {task_name}")
    dataset_args = loads(dataset_args)
    dataset_args["return_dict"] = True
    dataset_args["eval_frac"] = eval_frac
    
    if not no_wandb:
        import wandb
        wandb.init(project="flu",
                   entity="mikeamerrill",
                   notes=notes)
        # wandb.log({"train_class_balance":train_class_balance})                   

    task = get_task_with_name(task_name)(dataset_args=dataset_args)
    
    train_dataset = task.get_train_dataset()
    eval_dataset = task.get_eval_dataset()

    infer_example = train_dataset[0]["inputs_embeds"]
    n_timesteps, n_features = infer_example.shape

    config = BertConfig(hidden_size=n_features,
                        num_attention_heads=num_attention_heads,
                        num_hidden_layers=num_hidden_layers,
                        max_position_embeddings=max_position_embeddings)
    
    
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=n_epochs,              # total # of training epochs
        per_device_train_batch_size=train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=eval_batch_size,   # batch size for evaluation
        warmup_steps=warmup_steps,                # number of warmup steps for learning rate scheduler
        weight_decay=weight_decay,               # strength of weight decay
        logging_dir='./logs',
        do_eval=True,
        prediction_loss_only=False,
        evaluation_strategy="epoch",
        report_to=["wandb"]            # directory for storing logs
    )

    model = BertForSequenceClassification(config)
    model.cuda()
    
    metrics = task.get_huggingface_metrics(threshold=classification_threshold)
    trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=metrics)
    trainer.train()
    trainer.evaluate()