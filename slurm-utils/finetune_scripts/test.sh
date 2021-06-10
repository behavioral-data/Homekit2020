python ./slurm-utils/slurm-launch.py \
        --dir "/homes/gws/mikeam/seattleflustudy"\
        --exp-name "test_1" \
        --num-nodes 1 \
        --conda-path "/gscratch/bdata/mikeam/anaconda3/bin/"\
        --account "bdata" \
        --partition "bdata-gpu" \
        --num-gpus 4 \
        --conda-env seattleflustudy \
        --command "python ./slurm-utils/raytune-test.py"
