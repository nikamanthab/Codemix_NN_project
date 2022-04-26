#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --ntasks=16
#SBATCH --mem=16gb
#SBATCH --job-name=nnproj
#SBATCH --output=../outputs/fasttext_offenseval_dravidian_ml
#SBATCH --account=PAS2168 

set -x

source activate ml
cd /users/PAS2056/appiahbalaji2/courses/nn/codemix/program
python run.py \
--weights_path /users/PAS2056/appiahbalaji2/courses/nn/codemix/weights/ \
--model_name fasttext \
--type magnitude \
--dataset offenseval_dravidian_ml