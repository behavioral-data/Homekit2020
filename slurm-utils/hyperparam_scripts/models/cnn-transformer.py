import os
os.environ["DEBUG_DATA"] = "1"
os.environ["WANDB_MODE"] = "offline"
#os.environ["WANDB_SILENT"] = "true"
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
def train_fn(config,checkpoint_dir=None):
    dataset_args = validate_dataset_args(None,None,"/gscratch/bdata/mikeam/SeattleFluStudy/src/data/dataset_configs/PredictTrigger.yaml")
    train_cnn_transformer("PredictTrigger",
                          dataset_args=dataset_args,
                          train_batch_size=500,
                          eval_batch_size=500, 
                          warmup_steps=20,
                          loss_fn="FocalLoss",
 			  look_for_cached_datareader=True,                        
			  no_eval_during_training=True,
                          **config)

def main():
#    obj_ref = ray.put(pickle.load(open("/gscratch/bdata/mikeam/SeattleFluStudy/data/processed/cached_datareaders/PredictTrigger-train_eval.pickle", "rb" )))
    analysis = tune.run(
        train_fn,
        num_samples=10,
        config={
            # define search space here
            "focal_alpha":tune.uniform(0.1,1.0),
            "focal_alpha":tune.uniform(1,3),
#            "datareader_ray_obj_ref":obj_ref,
            "learning_rate":tune.loguniform(1e-5, 1e-2),
            "num_attention_heads":tune.choice([1,2,4]),
            "num_hidden_layers":tune.choice([1,2,3,4]),
            # wandb configuration
            "wandb": {
                "project": "test",
                "api_key":"4175520a5f76bf971932f02a8ed7c9c4b2b38207"
            }
        },
        resources_per_trial={"gpu": 4},
	name="CNNTransformer-PredictTrigger",
	local_dir="/gscratch/bdata/mikeam/SeattleFluStudy/results")


    print("Best config: ", analysis.get_best_config(
        metric="eval_roc_auc", mode="min"))
    df = analysis.results_df
    df_path = os.path.join(__file__,"test.csv")
    print(f"Writing all results to {df_path}")
    df.to_csv(df_path)

if __name__ == "__main__":
    main()
