#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:0:10
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --output=./log/conventional.out
#SBATCH --error=./log/conventional.err

cd "${SLURM_SUBMIT_DIR}"

echo JOB ID: "${SLURM_JOBID}"

echo Working Directory: $(pwd)

# module add lang/python/anaconda/pytorch

echo Start Time: $(date)

time python3 ./run_plms.py --batch_size 8 --do_train True --epoch 15 --pretrained_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --learning_rate 1e-4 --weight_decay 1e-4 --model_dir models/coventional --model_name multiclass-abstract-${} --multi_class True --train_data_dir "dl_data/desc/CKS only"

echo End Time: $(date)