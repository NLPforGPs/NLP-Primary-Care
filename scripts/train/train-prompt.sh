#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:0:10
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --output=./log/prompt.out
#SBATCH --error=./log/prompt.err

cd "${SLURM_SUBMIT_DIR}"

echo JOB ID: "${SLURM_JOBID}"

echo Working Directory: $(pwd)

# module add lang/python/anaconda/pytorch

echo Start Time: $(date)

time python3 ./run_plms.py --batch_size 8 --do_train True --epoch 15 --pretrained_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract --learning_rate 1e-4 --weight_decay 1e-4 --use_prompt True --model_dir models/prompt --model_name multiclass-abstract-optimal --multi_class True --train_data_dir "dl_data/desc/CKS only" --prompt "This is a problem of {}."

echo End Time: $(date)
