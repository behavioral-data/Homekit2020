PRETRAIN_LOC="/gscratch/bdata/mikeam/SeattleFluStudy/models/pretrained_on_all_flu_neg.ckpt"
BASE_COMMAND="python src/__main__.py train-cnn-transformer --train_batch_size 700 --dropout_rate=0.4 --learning_rate=5e-5 --n_epochs=20 --warmup_steps=20 --val_epochs 2 --model_config model_configs/small_embedding.yaml --model_path $PRETRAIN_LOC --task_config ./src/data/task_configs/PredictFluPos.yaml"

for i in $(seq 0 9); do  
        pythonCommand="$BASE_COMMAND --train_path data/processed/flu_pos_folds/f${i}_train/ --eval_path data/processed/flu_pos_folds/f${i}_test/ --notes 'Fold ${i}'"
        python slurm-utils/launch_on_slurm.py  -p \"gpu-2080ti\" -n 1 -m '32G' --num-gpus 1 --num-cpus 5 --dir . --exp-name flu_pos_fold --command \"$pythonCommand\"
    done