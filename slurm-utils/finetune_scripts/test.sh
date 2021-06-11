python ./slurm-utils/slurm-launch.py \
        --dir $PWD\
	--exp-name "test_1" \
        --num-nodes 1 \
        --conda-path "/gscratch/bdata/mikeam/anaconda3"\
        --account "cse" \
        --partition "cse-gpu" \
        --num-gpus 6 \
        --conda-env seattleflustudy \
        --command "python ./slurm-utils/raytune-test.py"
