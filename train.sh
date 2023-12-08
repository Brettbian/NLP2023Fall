#!/bin/bash
python finetune.py --dataset_name BENBENBENb/RACE1000COT --output_dir brettbbb/race_cot_16 --epoch 20 --cot --train_size 16

python finetune.py --dataset_name BENBENBENb/RACE1000COT --output_dir brettbbb/race_cot_32 --epoch 20 --cot --train_size 32

python finetune.py --dataset_name BENBENBENb/RACE1000COT --output_dir brettbbb/race_cot_64 --epoch 20 --cot --train_size 64

python finetune.py --dataset_name BENBENBENb/RACE1000COT --output_dir brettbbb/race_cot_128 --epoch 20 --cot --train_size 128

python finetune.py --dataset_name BENBENBENb/RACE1000COT --output_dir brettbbb/race_cot_128 --epoch 20 --cot --train_size 256