TRAIN_PATH="/mmfs1/gscratch/bdata/mikeam/MobileSensingSuite/data/processed/audere/audere_split_2020_02_10_to_2020_04_01_by_participant/train_7_day_no_scale"
EVAL_PATH="/mmfs1/gscratch/bdata/mikeam/MobileSensingSuite/data/processed/audere/audere_split_2020_02_10_to_2020_04_01_by_participant/eval_7_day_no_scale"
TEST_PATH="/mmfs1/gscratch/bdata/mikeam/MobileSensingSuite/data/processed/audere/audere_split_2020_02_10_to_2020_04_01_by_participant/test_7_day_no_scale"
PRETRAINED_PATH="/mmfs1/gscratch/bdata/mikeam/MobileSensingSuite/models/pretrained_CNNTransformer_predict_daily_features_participant_split.ckpt"
BASE_COMMAND="--config configs/models/CNNToTransformerClassifier.yaml --model.pretrained_ckpt_path $PRETRAINED_PATH --model.dropout_rate 0.4  --model.learning_rate 0.003  --data.train_path $TRAIN_PATH --data.val_path $EVAL_PATH --data.test_path $TEST_PATH --model.batch_size 800 --trainer.check_val_every_n_epoch 1 --trainer.max_epochs 50 --trainer.log_every_n_steps 50 --early_stopping_patience 40 --model.warmup_steps 100 "

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
    eval "python slurm-utils/launch_on_slurm.py  -n 1 -m '64G' --num-gpus 1 --num-cpus 5 --dir . --exp-name Split-CNNTransformer-${TASKS[$i]} --command \"$pythonCommand\""
  done