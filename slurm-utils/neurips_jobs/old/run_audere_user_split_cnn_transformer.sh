TRAIN_PATH="/gscratch/bdata/estebans/Homekit2020/data/processed/split/audere_split_2020_02_10/train_7_day"
EVAL_PATH="/gscratch/bdata/estebans/Homekit2020/data/processed/split/audere_split_2020_02_10/eval_7_day"
TEST_PATH="/gscratch/bdata/estebans/Homekit2020/data/processed/split/audere_split_2020_02_10/test_7_day"


BASE_COMMAND="--config configs/models/CNNToTransformerClassifier.yaml --model.learning_rate 0.003 --data.test_path $TEST_PATH --data.train_path $TRAIN_PATH --data.val_path $EVAL_PATH --model.batch_size 800 --trainer.check_val_every_n_epoch 1 --trainer.max_epochs 50 --trainer.log_every_n_steps 50 --early_stopping_patience 10 --checkpoint_metric  val/roc_auc"

wandb artifact cache cleanup 1GB

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
      pythonCommand="python src/models/train.py fit --config configs/tasks/${TASKS[$i]}.yaml ${BASE_COMMAND} --run_name ${TASKS[$i]}-seed_${seed}-$(date +%F) --notes 'Temporal Split'"
#      eval $pythonCommand
      eval "python slurm-utils/launch_on_slurm.py  -n 1 -m '48G' --num-gpus 1 -p gpu-rtx6k --num-cpus 5 --dir . --exp-name CNNTransformer-${TASKS[$i]}-temp_split-seed_${seed} --command \"$pythonCommand\" --conda-env \"mobs\""
    done
  done
