import os
#os.environ["DEBUG_DATA"] = "1"
os.environ["WANDB_MODE"] = "offline"
#os.environ["WANDB_SILENT"] = "true"
#os.environ["TRANSFORMERS_VERBOSITY"] = "error"
#os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
import pickle

import ray
from ray.tune.integration.wandb import wandb_mixin
from ray import tune
from src.models.train_model import train_cnn_transformer
from src.models.commands import validate_dataset_args
from src.utils import get_logger


def train_fn(config,checkpoint_dir=None):
    dataset_args = validate_dataset_args(None,None,"/gscratch/bdata/mikeam/SeattleFluStudy/src/data/dataset_configs/PredictFluPos.yaml")
    train_cnn_transformer("PredictFluPos",
                          dataset_args=dataset_args,
                          train_batch_size=500,
                          eval_batch_size=500, 
                          warmup_steps=20,
                          loss_fn="FocalLoss",
# 			  look_for_cached_datareader=True, 
                          task_ray_obj_ref = config["obj_ref"],                    
			              no_eval_during_training=True,
                          output_dir = tune.get_trial_dir(),
                          tune=True,
			              no_wandb=True,
                          **config)

def main():
    print(os.environ["ip_head"], os.environ["redis_password"])
    ray.init(address='auto', _node_ip_address=os.environ["ip_head"].split(":")[0], _redis_password=os.environ["redis_password"])
    obj_ref = ray.put(pickle.load(open("/gscratch/bdata/mikeam/SeattleFluStudy/data/processed/cached_tasks/PredictTrigger-train-eval.pickle", "rb" )))
    analysis = tune.run(
        train_fn,
        num_samples=25,
        config={
            # define search space here
            "focal_alpha":tune.uniform(0.1,1.0),
            "focal_gamma":tune.uniform(1,3),
            "obj_ref":obj_ref,
            "learning_rate":tune.loguniform(1e-6,1e-4),
            "n_epochs":tune.choice(range(10,50)),
            "num_hidden_layers":0,
            # wandb configuration
            "wandb": {
                "project": "test",
                "api_key":"4175520a5f76bf971932f02a8ed7c9c4b2b38207"
            }
        },
        resources_per_trial={"gpu": 4},
	name="CNN-PredictFluPos",
	local_dir="/gscratch/bdata/mikeam/SeattleFluStudy/results")

    df = analysis.results_df
    print("Best config: ", analysis.get_best_config(
        metric="eval/roc_auc", mode="min"))
    
    df_path = os.path.join(analysis._experiment_dir,"CNN-PredictFluPos.csv")
    print(f"Writing all results to {df_path}")
    df.to_csv(df_path)

    best_score = df["eval/roc_auc"].min()
    print(f"Best Score: {best_score}")
    

if __name__ == "__main__":
    main()
