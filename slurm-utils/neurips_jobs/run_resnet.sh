TRAIN_PATH="/mmfs1/gscratch/bdata/mikeam/MobileSensingSuite/data/processed/COVID_Signals/train_7_day_no_scale_public_only"
EVAL_PATH="/mmfs1/gscratch/bdata/mikeam/MobileSensingSuite/data/processed/COVID_Signals/eval_7_day_no_scale_public_only"
BASE_COMMAND="--model ResNet --model.batch_size 64  --model.dropout_rate 0.4  --model.learning_rate 0.003  --data.train_path $TRAIN_PATH --data.val_path $EVAL_PATH --model.batch_size 800 --trainer.check_val_every_n_epoch 1 --trainer.max_epochs 50 --trainer.log_every_n_steps 50 --early_stopping_patience 40 --model.warmup_steps 100"

TASKS=(
    "CovidSignalsPositivity"
    "CovidSignalsAches"
    "CovidSignalsChills"
    "CovidSignalsHeadache"
    "CovidSignalsCough"
)

TASK_ARGS=(
    ""
    "--data.survey_path data/processed/COVID_Signals/daily_surveys.csv"
    "--data.survey_path data/processed/COVID_Signals/daily_surveys.csv"
    "--data.survey_path data/processed/COVID_Signals/daily_surveys.csv"
    "--data.survey_path data/processed/COVID_Signals/daily_surveys.csv"
)
for i in ${!TASKS[*]}; 
  do
    pythonCommand="python src/models/train.py fit --config configs/tasks/${TASKS[$i]}.yaml $BASE_COMMAND  ${TASK_ARGS[$i]}"
    eval "python slurm-utils/launch_on_slurm.py  -n 1 -m '64G' --num-gpus 1 --num-cpus 5 --dir . --exp-name ${TASKS[$i]} --command \"$pythonCommand\""
  done
