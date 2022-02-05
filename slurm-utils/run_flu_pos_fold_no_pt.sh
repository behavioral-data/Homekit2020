BASE_COMMAND="python src/__main__.py train-cnn-transformer --train_batch_size 700 --dropout_rate=0.4 --learning_rate=5e-4 --n_epochs=300 --warmup_steps=50 --val_epochs 10  --model_config model_configs/small_embedding.yaml --task_config ./src/data/task_configs/PredictFluPos.yaml"
for i in $(seq 0 19); do  
        pythonCommand="$BASE_COMMAND --train_path $PWD/data/processed/flu_pos_folds_20/f${i}_train/ --eval_path $PWD/data/processed/flu_pos_folds_20/f${i}_test/ --notes 'Fold ${i} - Not Pretrained'"
        python slurm-utils/launch_on_slurm.py  -p \"ckpt\" -n 1 -m '32G' --num-gpus 1 --num-cpus 5 --dir . --exp-name flu_pos_fold_${i} --command $pythonCommand
    done