#!/bin/bash

## !!! Update the paths in the extender.py script to point to the correct data directories before running this script !!

cd ../src
python extender.py \
    --num_devices 8 \
    --batch_size 16 \
    --max_epochs 400