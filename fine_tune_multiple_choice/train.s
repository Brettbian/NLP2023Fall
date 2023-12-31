#!/bin/bash
#
#SBATCH --job-name=arc_tt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --mem=15GB
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=yb970@nyu.edu

# Execute your commands inside a Singularity container
singularity exec --bind /scratch --nv --overlay /scratch/yb970/capstone/overlay-25GB-500K.ext3:ro \
  /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash -c '

# Source the environment script within the Singularity container
source /ext3/env.sh

# Activate the conda environment
conda activate

# Your GPU-accelerated command (replace this with your actual command)
cd /scratch/yb970/NLP2023Fall

python finetune.py --dataset_name BENBENBENb/ARC1000COT --output_dir brettbbb/arc_cot_16 --epoch 20 --cot --train_size 16 --skip_example
echo "16 done"
python finetune.py --dataset_name BENBENBENb/ARC1000COT --output_dir brettbbb/arc_cot_32 --epoch 20 --cot --train_size 32 --skip_example
echo "32 done"
python finetune.py --dataset_name BENBENBENb/ARC1000COT --output_dir brettbbb/arc_cot_64 --epoch 20 --cot --train_size 64 --skip_example
echo "64 done"
python finetune.py --dataset_name BENBENBENb/ARC1000COT --output_dir brettbbb/arc_cot_128 --epoch 20 --cot --train_size 128 --skip_example
echo "128 done"
python finetune.py --dataset_name BENBENBENb/ARC1000COT --output_dir brettbbb/arc_cot_256 --epoch 20 --cot --train_size 256 --skip_example
echo "256 done"
python inference.py --finetuned_model brettbbb/arc_cot_16 --dataset_name BENBENBENb/ARC1000COT --output_dir brettbbb/arc_cot_16
python inference.py --finetuned_model brettbbb/arc_cot_32 --dataset_name BENBENBENb/ARC1000COT --output_dir brettbbb/arc_cot_32
python inference.py --finetuned_model brettbbb/arc_cot_64 --dataset_name BENBENBENb/ARC1000COT --output_dir brettbbb/arc_cot_64 
python inference.py --finetuned_model brettbbb/arc_cot_128 --dataset_name BENBENBENb/ARC1000COT --output_dir brettbbb/arc_cot_128
python inference.py --finetuned_model brettbbb/arc_cot_256 --dataset_name BENBENBENb/ARC1000COT --output_dir brettbbb/arc_cot_256
'

