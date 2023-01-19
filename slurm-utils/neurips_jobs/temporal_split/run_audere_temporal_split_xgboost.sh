TRAIN_PATH="/gscratch/bdata/estebans/Homekit2020/data/processed/hourly_user_split/train_7_day_hourly"
EVAL_PATH="/gscratch/bdata/estebans/Homekit2020/data/processed/hourly_user_split/eval_7_day_hourly"
TEST_PATH="/gscratch/bdata/estebans/Homekit2020/data/processed/hourly_user_split/test_7_day_hourly"

export WANDB_DISABLE_SERVICE=True
export WANDB_CACHE_DIR="/gscratch/bdata/estebans/Homekit2020/.wandb_cache"
wandb artifact cache cleanup 1GB

BASE_COMMAND="--config configs/models/XGBoostClassifier.yaml --model.early_stopping_rounds 10 --data.test_path $TEST_PATH --data.train_path $TRAIN_PATH --data.val_path $EVAL_PATH --checkpoint_metric  val/roc_auc"

TASKS=(
    "HomekitPredictFluPos"
    "HomekitPredictFluSymptoms"
    "HomekitPredictSevereFever"
    "HomekitPredictCough"
    "HomekitPredictFatigue"
)

SEEDS=(0 1 2 3 4)

for i in ${!TASKS[*]};
  do
    for seed in ${!SEEDS[*]};
    do
      EXPERIMENT_NAME="XGBoost-${TASKS[$i]}-temp_split-seed_${seed}-$(date +%F)"
      pythonCommand="python src/models/train.py fit --config configs/tasks/${TASKS[$i]}.yaml ${BASE_COMMAND}  --pl_seed ${seed} --run_name ${EXPERIMENT_NAME} --no_wandb --notes 'temporal split'"
      eval $pythonCommand
      exit
#      eval "python slurm-utils/launch_on_slurm.py  -n 1 -m '36G' --num-gpus 1 -p gpu-rtx6k --num-cpus 4 --dir . --exp-name ${EXPERIMENT_NAME} --command \"$pythonCommand\" --conda-env \"mobs\""
    done
  done