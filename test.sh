#!/bin/bash

#BSUB -J 0_0
#BSUB -q gpu_v100
#BSUB -gpu "num=2:mode=exclusive_process:aff=yes"


#python demo.py
python -m torch.distributed.launch --nproc_per_node=2 evaluate.py

