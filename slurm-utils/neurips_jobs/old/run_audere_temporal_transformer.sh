TRAIN_PATH="/gscratch/bdata/mikeam/MobileSensingSuite/data/processed/audere/audere_split_2020_02_10_to_2020_04_01/train_7_day_no_scale_hour_level/"
EVAL_PATH="/gscratch/bdata/mikeam/MobileSensingSuite/data/processed/audere/audere_split_2020_02_10_to_2020_04_01/eval_7_day_no_scale_hour_level/"
TEST_PATH="/gscratch/bdata/mikeam/MobileSensingSuite/data/processed/audere/audere_split_2020_02_10_to_2020_04_01/test_7_day_no_scale_hour_level/"
BASE_COMMAND="--model src.models.models.TransformerClassifier --model.num_hidden_layers 9 --model.learning_rate 0.002  --data.train_path $TRAIN_PATH --data.val_path $EVAL_PATH  --data.test_path $TEST_PATH --model.batch_size 400 --trainer.check_val_every_n_epoch 1 --trainer.max_epochs 50 --trainer.log_every_n_steps 50 --early_stopping_patience 3 --model.warmup_steps 250"

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
    eval "python slurm-utils/launch_on_slurm.py  -n 1 -m '64G' --num-gpus 1 --num-cpus 5 --dir . --exp-name ResNet-${TASKS[$i]} --command \"$pythonCommand\""
  done