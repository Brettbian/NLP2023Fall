#!/bin/bash
srun --pty -c 2 --mem=15GB -t2:00:00 --gres=gpu:v100:1 /bin/bash

singularity exec --bind /scratch --nv --overlay /scratch/yb970/capstone/overlay-25GB-500K.ext3:ro \
  /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash

source /ext3/env.sh

conda activate

cd /scratch/yb970/NLP2023Fall

python finetune.py --dataset_name BENBENBENb/RACE1000COT --output_dir brettbbb/race_cot_16 --epoch 20 --cot --train_size 16

python finetune.py --dataset_name BENBENBENb/RACE1000COT --output_dir brettbbb/race_cot_32 --epoch 20 --cot --train_size 32

python finetune.py --dataset_name BENBENBENb/RACE1000COT --output_dir brettbbb/race_cot_64 --epoch 20 --cot --train_size 64

python finetune.py --dataset_name BENBENBENb/RACE1000COT --output_dir brettbbb/race_cot_128 --epoch 20 --cot --train_size 128

python finetune.py --dataset_name BENBENBENb/RACE1000COT --output_dir brettbbb/race_cot_128 --epoch 20 --cot --train_size 256