TRAIN_PATH="/gscratch/bdata/mikeam/SeattleFluStudy/data/processed/split/audere_split_2020_02_10_to_2020_04_01/train_7_day_no_scale/"
EVAL_PATH="/gscratch/bdata/mikeam/SeattleFluStudy/data/processed/split/audere_split_2020_02_10_to_2020_04_01/eval_7_day_no_scale/"
TEST_PATH="/gscratch/bdata/mikeam/SeattleFluStudy/data/processed/split/audere_split_2020_02_10_to_2020_04_01/test_7_day_no_scale/"

wandb artifact cache cleanup 1GB

BASE_COMMAND="--config configs/models/CNNToTransformerClassifier.yaml --model.learning_rate 0.003 --data.test_path $TEST_PATH --data.train_path $TRAIN_PATH --data.val_path $EVAL_PATH --model.batch_size 800 --trainer.check_val_every_n_epoch 1 --trainer.max_epochs 50 --trainer.log_every_n_steps 50 --early_stopping_patience 10 --checkpoint_metric  val/roc_auc"
BASE_COMMAND="--config configs/models/CNNToTransformerClassifier.yaml --model.learning_rate 0.003 --data.test_path $TEST_PATH --data.train_path $TRAIN_PATH --data.val_path $EVAL_PATH --model.batch_size 800 --trainer.check_val_every_n_epoch 1 --trainer.max_epochs 50 --trainer.log_every_n_steps 50 --early_stopping_patience 10  --checkpoint_metric  val/roc_auc"


TASKS=(
    "HomekitPredictFluPos"
#    "HomekitPredictFluSymptoms"
#    "HomekitPredictSevereFever"
#    "HomekitPredictCough"
#    "HomekitPredictFatigue"
)

SEEDS=(0 1 2 3 4)
SEEDS=(0)


for i in ${!TASKS[*]};
  do
    for seed in ${!SEEDS[*]};
    do
      EXPERIMENT_NAME="Private-CNNTransformer-${TASKS[$i]}-temp_split-seed_${seed}-$(date +%F)"

      pythonCommand="python src/models/train.py fit --config configs/tasks/${TASKS[$i]}.yaml ${BASE_COMMAND} --no_wandb --run_name ${EXPERIMENT_NAME} --notes 'USER Split'"
      eval $pythonCommand
#      eval "python slurm-utils/launch_on_slurm.py  -n 1 -m '36G' --num-gpus 1 -p gpu-rtx6k --num-cpus 4 --dir . --exp-name ${EXPERIMENT_NAME} --command \"$pythonCommand\" --conda-env \"mobs\""
    done
  done
