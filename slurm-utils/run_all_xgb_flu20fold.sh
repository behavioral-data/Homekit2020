DATA_PATH="/homes/gws/mikeam/seattleflustudy/data/processed/flu_pos_folds_20"
BASE_COMMAND="python src/__main__.py train-xgboost --task_config ./src/data/task_configs/PredictFluPos.yaml "
for i in $(seq 0 19); do  
        pythonCommand="$BASE_COMMAND --train_participant_dates ${DATA_PATH}/f${i}_train_peds.yaml  --eval_participant_dates ${DATA_PATH}/f${i}_test_peds.yaml" 
        eval $pythonCommand
    done