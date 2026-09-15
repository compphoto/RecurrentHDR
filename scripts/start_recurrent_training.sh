#!/bin/bash

## !!! Update the paths in the recurrent_extender.py script to point to the correct data directories before running this script !!

cd ../src
python recurrent_extender.py \
    --num_devices 8 \
    --batch_size 16 \
    --max_epochs 500 \
    --stages 2 \
    --resume \
    --last_run_dir  /path/to/single/stage/checkpoint \
 