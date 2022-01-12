TRAIN_PATH="/gscratch/bdata/mikeam/SeattleFluStudy/data/processed/split/audere_split_2020_02_10_to_2020_04_01/train_7_day_no_scale/"
EVAL_PATH="/gscratch/bdata/mikeam/SeattleFluStudy/data/processed/split/audere_split_2020_02_10_to_2020_04_01/eval_7_day_no_scale/"
TEST_PATH="/gscratch/bdata/mikeam/SeattleFluStudy/data/processed/split/audere_split_2020_02_10_to_2020_04_01/test_7_day_no_scale/"
BASE_COMMAND="python src/__main__.py train-cnn-transformer --train_batch_size 700 --dropout_rate=0.4 --focal_gamma=1.7188155817156203 --learning_rate=5e-5 --n_epochs=50 --warmup_steps=20 --val_epochs 2 --model_config model_configs/small_embedding.yaml --train_path $TRAIN_PATH --eval_path $EVAL_PATH --test_path $TEST_PATH --early_stopping"
TASKS=(
    "PredictFluPos"
    "PredictFluSymptoms"
    "PredictMobilityDifficulty"
    "PredictWeekend"
    "PredictSevereSymptoms"
    "PredictTrigger"
    "PredictCough"
    "PredictFatigue"
)

for task in ${TASKS[*]} 
  do
    pythonCommand="$BASE_COMMAND --task_config ./src/data/task_configs/$task.yaml"
    eval "python slurm-utils/launch_on_slurm.py  -n 1 -m '64G' --num-gpus 2 --num-cpus 10 --dir . --exp-name $task --command \"$pythonCommand\""
  done
