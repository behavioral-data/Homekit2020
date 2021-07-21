import os
os.environ["DEBUG_DATA"] = "1"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
import pickle

import ray
from ray.tune.integration.wandb import wandb_mixin
from ray import tune
from src.models.train_model import train_cnn_transformer
from src.models.commands import validate_dataset_args
from src.utils import get_logger


@wandb_mixin
def train_fn(config, checkpoint_dir=None):
    dataset_args = validate_dataset_args(None,None,"/homes/gws/mikeam/seattleflustudy/src/data/dataset_configs/PredictTrigger.yaml")
    train_cnn_transformer("PredictTrigger",
                          dataset_args=dataset_args,
                          train_batch_size=300,
                          task_ray_obj_ref=config["obj_ref"],
                          eval_batch_size=400,
                          n_epochs=1,
                          tune=True,
                          output_dir=checkpoint_dir,
                          loss_fn="CrossEntropyLoss",
                          no_wandb=True)

print("loading_pickle")
ray.init(_memory=6e10,object_store_memory=5e10)
obj_ref = ray.put(pickle.load(open("/homes/gws/mikeam/seattleflustudy/data/debug/cached_tasks/PredictTrigger-train-eval.pickle", "rb" )))
analysis = tune.run(
    train_fn,
    num_samples=1,
    config={
        # define search space here
        "gpu":4,
        "num_samples":20,
        "num_gpus_per_worker":1,
        "warmup_steps": tune.choice([10, 40]),
        "pos_class_weight": tune.uniform(0, 10),
        "obj_ref":obj_ref,
        # wandb configuration
        "wandb": {
            "project": "test",
            "api_key":"4175520a5f76bf971932f02a8ed7c9c4b2b38207"
        }
    },
    resources_per_trial={"gpu": 2})


print("Best config: ", analysis.get_best_config(
    metric="eval/roc_auc", mode="min"))
