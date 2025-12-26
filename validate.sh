#!/bin/bash
# Initialize conda (adjust path if needed)
source ~/anaconda3/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh || eval "$(conda shell.bash hook)"

conda activate autockt
export PATH=$HOME/ngspice-27/opt/bin:$PATH
export PYTHONPATH=$PWD
export RAY_TMPDIR=$HOME/AutoCkt/ray_tmp
python autockt/gen_specs.py --num_specs 100
# let CHECKPOINT_PTH be the arg of the script
CHECKPOINT_PTH=$1
ipython -c "%run autockt/rollout.py $CHECKPOINT_PTH --run PPO --env opamp-v0 --num_val_specs 100 --traj_len 30 --no-render"