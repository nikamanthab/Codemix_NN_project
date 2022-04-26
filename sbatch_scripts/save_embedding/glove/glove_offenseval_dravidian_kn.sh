#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16gb
#SBATCH --job-name=nnproj
#SBATCH --output=../outputs/glove_offenseval_dravidian_kn
#SBATCH --account=PAS2168 

set -x

source activate ml
cd /users/PAS2056/appiahbalaji2/courses/nn/codemix/program
python saveEmbedding.py \
--weights_path /users/PAS2056/appiahbalaji2/courses/nn/codemix/weights/ \
--model_name glove \
--type magnitude \
--dataset offenseval_dravidian_kn