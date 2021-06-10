import os
os.environ["DEBUG_DATA"] = "1"
import logging

from ray.tune.integration.wandb import wandb_mixin
from ray import tune
from src.models.train_model import train_cnn_transformer
from src.models.commands import validate_dataset_args
from src.utils import get_logger

@wandb_mixin
def train_fn(config):
    
    dataset_args = validate_dataset_args(None,None,"/homes/gws/mikeam/seattleflustudy/src/data/dataset_configs/PredictTrigger.yaml")
    train_cnn_transformer("PredictTrigger",
                          dataset_args=dataset_args,
                          train_batch_size=300,
                          look_for_cached_datareader=True,
                          eval_batch_size=400,
                          loss_fn="CrossEntropyLoss")

# logging.basicConfig(force=True)
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.WARNING)

tune.run(
    train_fn,
    config={
        # define search space here
        "gpu":1,
        "num_gpus_per_worker":1,
        "warmup_steps": tune.choice([10, 40]),
        # wandb configuration
        "wandb": {
            "project": "test",
            "api_key":"4175520a5f76bf971932f02a8ed7c9c4b2b38207"
        }
    },
    resources_per_trial={"gpu": 1})
