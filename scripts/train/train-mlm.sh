#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:20:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --output=./log/train/mlm.out
#SBATCH --error=./log/train/mlm.err

cd "${SLURM_SUBMIT_DIR}"

echo JOB ID: "${SLURM_JOBID}"

echo Working Directory: $(pwd)

module add lang/python/anaconda/pytorch
# module add lang/python/anaconda/3.8-2020.07

conda env create --file nlp_gp.yml

eval "$(conda shell.bash hook)"
conda activate NLP_GP

# module add lang/python/anaconda/pytorch

echo Start Time: $(date)

time python3 ./run_plms.py --batch_size 8 --epoch 15 --pretrained_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract --learning_rate 1e-4 --weight_decay 1e-4 --model_dir models/mlm --model_name multiclass-abstract --train_data_dir "dl_data/desc/Ccks_only" --prompt "This is a problem of {}."  --do_train --use_mlm

echo End Time: $(date)
