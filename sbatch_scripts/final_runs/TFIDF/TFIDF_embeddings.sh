#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16gb
#SBATCH --job-name=nnproj
#SBATCH --output=../outputs/TFIDF_kan_hope
#SBATCH --account=PAS2168 

set -x

source activate nnproj
cd /users/PAS2056/appiahbalaji2/courses/nn/codemix/program
python run_vectorizer.py \
--model_name TfidfVectorizer \
--type TfidfVectorizer \
--dataset kan_hope