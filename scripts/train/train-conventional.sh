#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --output=./log/train/train-conventional.out
#SBATCH --error=./log/train/train-conventional.err

cd "${SLURM_SUBMIT_DIR}"

echo JOB ID: "${SLURM_JOBID}"

echo Working Directory: $(pwd)

module add lang/python/anaconda/pytorch
# module add lang/python/anaconda/3.8-2020.07

conda env create --file nlp_gp.yml

eval "$(conda shell.bash hook)"
conda activate NLP_GP

 
echo Start Time: $(date)

time python3 ./run_plms.py --batch_size 8 --epoch 10 --pretrained_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --learning_rate 5e-5 --weight_decay 1e-4 --model_dir models/conventional --model_name full-text-conventional --label_path label2id.json --prompt "This is a problem of {}."  --do_train

echo End Time: $(date)