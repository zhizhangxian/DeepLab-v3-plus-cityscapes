#!/bin/bash

#BSUB -J cityscapes_baseline
#BSUB -q gpu_v100
#BSUB -gpu "num=2:mode=exclusive_process:aff=yes"

python -m torch.distributed.launch --nproc_per_node=2 demo.py

