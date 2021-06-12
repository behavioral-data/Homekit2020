python ./slurm-utils/slurm-launch.py \
        --dir $PWD\
	    --exp-name "$1" \
        --num-nodes 2 \
        --conda-path "/gscratch/bdata/mikeam/anaconda3"\
        --account "cse" \
        --partition "cse-gpu" \
        --num-gpus 8 \
        --conda-env seattleflustudy \
        --command "python $2"
