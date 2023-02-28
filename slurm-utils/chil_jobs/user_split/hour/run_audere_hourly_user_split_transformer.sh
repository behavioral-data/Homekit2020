TRAIN_PATH="/gscratch/bdata/datasets/homekit2020-1.0/split_2020_02_10/train_7_day_hourly"
EVAL_PATH="/gscratch/bdata/datasets/homekit2020-1.0/split_2020_02_10/eval_7_day_hourly"
TEST_PATH="/gscratch/bdata/datasets/homekit2020-1.0/split_2020_02_10/test_7_day_hourly"

export WANDB_DISABLE_SERVICE=True
export WANDB_CACHE_DIR="/gscratch/bdata/estebans/Homekit2020/.wandb_cache"
wandb artifact cache cleanup 1GB

BASE_COMMAND="--config configs/models/TransformerClassifier.yaml --model.learning_rate 0.003 --data.test_path $TEST_PATH --data.train_path $TRAIN_PATH --data.val_path $EVAL_PATH --model.batch_size 800 --trainer.check_val_every_n_epoch 1 --trainer.max_epochs 50 --trainer.log_every_n_steps 50 --early_stopping_patience 10 --checkpoint_metric  val/roc_auc"

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
      EXPERIMENT_NAME="Transformer-${TASKS[$i]}-hourly_user_split-seed_${seed}-$(date +%F)"
      pythonCommand="python src/models/train.py fit --config configs/tasks/${TASKS[$i]}.yaml ${BASE_COMMAND} --pl_seed ${seed} --run_name ${EXPERIMENT_NAME} --notes 'hourly user split'"
      eval "python slurm-utils/launch_on_slurm.py  -n 1 -m '36G' --num-gpus 1 -p gpu-rtx6k --num-cpus 4 --dir . --exp-name ${EXPERIMENT_NAME} --command \"$pythonCommand\" --conda-env \"mobs\""
    done
  done