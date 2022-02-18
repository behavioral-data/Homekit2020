TRAIN_PARTICIPANTS_PATH="/projects/bdata/datasets/gatesfoundation/processed/split/audere_split_2020_02_10_to_2020_04_01/train_participant_dates.csv"
EVAL_PARTICIPANTS_PATH="/projects/bdata/datasets/gatesfoundation/processed/split/audere_split_2020_02_10_to_2020_04_01/eval_participant_dates.csv"
TEST_PARTICIPANTS_PATH="/projects/bdata/datasets/gatesfoundation/processed/split/audere_split_2020_02_10_to_2020_04_01/test_participant_dates.csv"
BASE_COMMAND="python src/__main__.py train-xgboost  --train_participant_dates $TRAIN_PARTICIPANTS_PATH --eval_participant_dates $EVAL_PARTICIPANTS_PATH --test_participant_dates $TEST_PARTICIPANTS_PATH"
TASKS=(
    "PredictFluPos"
    "PredictFluSymptoms"
    "PredictSevereFever"
    "PredictCough"
    "PredictFatigue"
)

for task in ${TASKS[*]} 
  do
    pythonCommand="$BASE_COMMAND --task_config ./src/data/task_configs/$task.yaml"
    echo "$pythonCommand"
  done
