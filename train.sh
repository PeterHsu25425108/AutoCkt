#!/bin/bash
# this is a bash script that sets environment variables
# make the changes persistent by adding them to your ~/.bashrc or ~/.bash_profile file

# Initialize conda (adjust path if needed)
source ~/anaconda3/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh || eval "$(conda shell.bash hook)"

conda activate autockt
python eval_engines/ngspice/ngspice_inputs/correct_inputs.py
python autockt/gen_specs.py --num_specs 570

export PATH=$HOME/ngspice-27/opt/bin:$PATH
export PYTHONPATH=$PWD
export RAY_TMPDIR=$HOME/AutoCkt/ray_tmp

# Run the main script in ipython by %run autockt/val_autobag_ray.py
ipython -c "%run autockt/val_autobag_ray.py"