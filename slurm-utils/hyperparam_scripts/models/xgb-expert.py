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
from src.models.baselines import train_xgboost
from src.models.commands import validate_yaml_or_json
from src.utils import get_logger


def train_fn(config,checkpoint_dir=None):
    dataset_args = validate_yaml_or_json(None,None,"/gscratch/bdata/mikeam/SeattleFluStudy/src/data/dataset_configs/PredictFatigue.yaml")
    train_xgboost("PredictSurveyClause",
                    dataset_args=dataset_args,
                    task_ray_obj_ref = config["obj_ref"],                    
                    tune=True,
                    no_wandb=True,
                    add_features_path="/gscratch/bdata/mikeam/SeattleFluStudy/data/processed/features/ExpertFeatures.csv",
                    **config)

def main():
    print(os.environ["ip_head"], os.environ["redis_password"])
    ray.init(address='auto', _node_ip_address=os.environ["ip_head"].split(":")[0], _redis_password=os.environ["redis_password"])
    obj_ref = ray.put(pickle.load(open("/gscratch/bdata/mikeam/SeattleFluStudy/data/processed/cached_tasks/PredictTrigger-Daily-train-eval.pickle", "rb" )))
    analysis = tune.run(
        train_fn,
        num_samples=25,
        config={
            # define search space here
            "max_depth":tune.choice(range(2,10)),
            "eta":tune.uniform(0.1,1),
            "obj_ref":obj_ref,
            # wandb configuration
            "wandb": {
                "project": "test",
                "api_key":"4175520a5f76bf971932f02a8ed7c9c4b2b38207"
            }
        },
        resources_per_trial={"gpu": 4},
	name="XGB-PredictFatigueExpert",
	local_dir="/gscratch/bdata/mikeam/SeattleFluStudy/results")


    print("Best config: ", analysis.get_best_config(
        metric="eval/roc_auc", mode="min"))
    df = analysis.results_df
    df_path = os.path.join(analysis._experiment_dir,"XGB-PredictFatigueExpert.csv")
    print(f"Writing all results to {df_path}")
    df.to_csv(df_path)

    best_score = df["eval/roc_auc"].max()
    print(f"Best Score: {best_score}")

if __name__ == "__main__":
    main()
