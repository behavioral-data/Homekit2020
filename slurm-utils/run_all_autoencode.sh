TRAIN_PATH="/gscratch/bdata/mikeam/SeattleFluStudy/data/processed/split/audere_split_2020_02_10_to_2020_04_01/train_7_day_no_scale/"
EVAL_PATH="/gscratch/bdata/mikeam/SeattleFluStudy/data/processed/split/audere_split_2020_02_10_to_2020_04_01/eval_7_day_no_scale/"

FLUVEY_TRAIN_PATH="/gscratch/bdata/mikeam/SeattleFluStudy/data/processed/split/fluvey_split_2021_02_15/train_7_day_no_scale/"
FLUVEY_EVAL_PATH="/gscratch/bdata/mikeam/SeattleFluStudy/data/processed/split/fluvey_split_2021_02_15/eval_7_day_no_scale/"

audere_command="python src/__main__.py train-cnn-transformer --task_config src/data/task_configs/Autoencode.yaml --n_epochs 100 --model_config ./model_configs/the_overfitter.yaml --learning_rate 0.001 --early_stopping --warmup_steps 337 --train_batch_size 200 --notes 'Audere Pretrain' --train_path $TRAIN_PATH --eval_path $EVAL_PATH  --val_epochs 5 --dropout_rate 0.1 --no_bootstrap"
fluvey_command="python src/__main__.py train-cnn-transformer --task_config src/data/task_configs/Autoencode.yaml --n_epochs 10 --model_config ./model_configs/the_overfitter.yaml --learning_rate 0.001 --early_stopping --warmup_steps 337 --train_batch_size 200 --notes 'Fluvey Pretrain' --train_path $FLUVEY_TRAIN_PATH --eval_path $FLUVEY_EVAL_PATH  --val_epochs 1 --dropout_rate 0.1 --no_bootstrap"
RESOURCES="python slurm-utils/launch_on_slurm.py  -n 1 -m '128G' --num-gpus 4 --num-cpus 14 --dir ."

echo "$RESOURCES --command \"$audere_command\" --exp-name 'audere-autoencode-pretrain'"
echo "$RESOURCES --command \"$fluvey_command\" --exp-name 'fluvey-autoencode-pretrain'"