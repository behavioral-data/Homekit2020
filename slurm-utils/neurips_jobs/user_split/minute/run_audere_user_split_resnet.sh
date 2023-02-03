TRAIN_PATH="/gscratch/bdata/datasets/homekit2020-1.0/split_2020_02_10_by_user/train_7_day"
EVAL_PATH="/gscratch/bdata/datasets/homekit2020-1.0/split_2020_02_10_by_user/eval_7_day"
TEST_PATH="/gscratch/bdata/datasets/homekit2020-1.0/split_2020_02_10_by_user/test_7_day"

export WANDB_DISABLE_SERVICE=True
export WANDB_CACHE_DIR="/gscratch/bdata/estebans/Homekit2020/.wandb_cache"
wandb artifact cache cleanup 1GB

BASE_COMMAND="--config configs/models/MyResNet.yaml --model.learning_rate 0.003 --data.test_path $TEST_PATH --data.train_path $TRAIN_PATH --data.val_path $EVAL_PATH --model.batch_size 320 --trainer.check_val_every_n_epoch 2 --trainer.max_epochs 50 --trainer.log_every_n_steps 50 --early_stopping_patience 5 --checkpoint_metric  val/roc_auc --model.val_bootstraps=0"

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
      EXPERIMENT_NAME="ResNet-${TASKS[$i]}-user_split-seed_${seed}-$(date +%F)"
      pythonCommand="python src/models/train.py fit --config configs/tasks/${TASKS[$i]}.yaml ${BASE_COMMAND} --pl_seed ${seed} --run_name ${EXPERIMENT_NAME} --notes 'user split'"
      eval "python slurm-utils/launch_on_slurm.py  -n 1 -m '64G' --num-gpus 1 -p gpu-a40 --num-cpus 6 --dir . --exp-name ${EXPERIMENT_NAME} --command \"$pythonCommand\" --conda-env \"Homekit2020\""
    done
  done