#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:20:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --output=./log/train-conventional.out
#SBATCH --error=./log/train-conventional.err

cd "${SLURM_SUBMIT_DIR}"

echo JOB ID: "${SLURM_JOBID}"

echo Working Directory: $(pwd)

module add lang/python/anaconda/pytorch
# module add lang/python/anaconda/3.8-2020.07

conda env create --file nlp_gp.yml

eval "$(conda shell.bash hook)"
conda activate NLP_GP


echo Start Time: $(date)

time python3 ./run_plms.py --batch_size 8 --epoch 15 --pretrained_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --learning_rate 1e-4 --weight_decay 1e-4 --model_dir models/conventional --model_name multiclass-fine-grained-conventional --train_data_dir "dl_data/desc/fine_grained" --prompt "This is a problem of {}."  --do_train --multi_class

echo End Time: $(date)