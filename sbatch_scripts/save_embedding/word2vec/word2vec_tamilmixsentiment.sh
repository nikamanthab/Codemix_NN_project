#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16gb
#SBATCH --job-name=nnproj
#SBATCH --output=../outputs/word2vec_tamilmixsentiment
#SBATCH --account=PAS2168 

set -x

source activate ml
cd /users/PAS2056/appiahbalaji2/courses/nn/codemix/program
python saveEmbedding.py \
--weights_path /users/PAS2056/appiahbalaji2/courses/nn/codemix/weights/ \
--model_name word2vec \
--type magnitude \
--dataset tamilmixsentiment