#!/bin/bash
TRAIN_PATH="/mmfs1/gscratch/bdata/mikeam/MobileSensingSuite/data/processed/COVID_Signals/train_7_day_no_scale_public_only"
EVAL_PATH="/mmfs1/gscratch/bdata/mikeam/MobileSensingSuite/data/processed/COVID_Signals/eval_7_day_no_scale_public_only"
BASE_COMMAND="--config configs/tasks/CovidSignalsPositivity.yaml --config configs/models/CNNToTransformerClassifier.yaml --model.dropout_rate 0.4  --model.learning_rate 0.003  --data.train_path $TRAIN_PATH --data.val_path $EVAL_PATH --model.batch_size 800 --trainer.check_val_every_n_epoch 1 --trainer.max_epochs 50 --trainer.log_every_n_steps 50 --early_stopping_patience 40 --model.warmup_steps 100"

TASK_ARGS=(
    "--data.window_onset_min 1 --data.window_onset_max 2"
    "--data.window_onset_min 1 --data.window_onset_max 3"
    "--data.window_onset_min 1 --data.window_onset_max 3"
    "--data.window_onset_max 2"
    "--data.window_onset_max 3"
    "--data.window_onset_max 4"
    "--data.window_onset_max 5"
)
for i in ${!TASK_ARGS[*]}; 
  do
    pythonCommand="python src/models/train.py fit $BASE_COMMAND  ${TASK_ARGS[$i]}"
    eval "python slurm-utils/launch_on_slurm.py  -n 1 -m '64G' --num-gpus 1 --num-cpus 5 --dir . --exp-name CovidSignalsWindowSize --command \"$pythonCommand\""
  done
