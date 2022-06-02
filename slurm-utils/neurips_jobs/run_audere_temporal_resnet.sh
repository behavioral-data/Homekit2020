TRAIN_PATH="/gscratch/bdata/mikeam/SeattleFluStudy/data/processed/split/audere_split_2020_02_10_to_2020_04_01/train_7_day_no_scale/"
EVAL_PATH="/gscratch/bdata/mikeam/SeattleFluStudy/data/processed/split/audere_split_2020_02_10_to_2020_04_01/eval_7_day_no_scale/"
TEST_PATH="/gscratch/bdata/mikeam/SeattleFluStudy/data/processed/split/audere_split_2020_02_10_to_2020_04_01/test_7_day_no_scale/"
BASE_COMMAND="--model src.models.models.ResNet --model.learning_rate 0.003  --data.train_path $TRAIN_PATH --data.val_path $EVAL_PATH  --data.test_path $TEST_PATH --model.batch_size 300 --trainer.check_val_every_n_epoch 1 --trainer.max_epochs 50 --trainer.log_every_n_steps 50 --early_stopping_patience 3 --model.warmup_steps 100"

TASKS=(
    "HomekitPredictFluPos"
    "HomekitPredictFluSymptoms"
    "HomekitPredictSevereFever"
    "HomekitPredictCough"
    "HomekitPredictFatigue"
)


for i in ${!TASKS[*]}; 
  do
    pythonCommand="python src/models/train.py fit --config configs/tasks/${TASKS[$i]}.yaml $BASE_COMMAND --notes 'User Split'"
    eval "python slurm-utils/launch_on_slurm.py  -n 1 -m '64G' --num-gpus 1 --num-cpus 5 --dir . --exp-name Split-ResNet-${TASKS[$i]} --command \"$pythonCommand\""
  done