#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:0:10
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --output=./log/prompt-test.out
#SBATCH --error=./log/prompt-test.err

cd "${SLURM_SUBMIT_DIR}"

echo JOB ID: "${SLURM_JOBID}"

echo Working Directory: $(pwd)

# module add lang/python/anaconda/pytorch

echo Start Time: $(date)

time python3 ./run_plms.py --batch_size 32 --do_predict True  --pretrained_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --load_checkpoint True --multi_class True --model_dir models/coventional --model_name multiclass-abstract-modified --prompt 'This is a problem of {}.' --predict_data_dir 'dl_data/transcripts'

echo End Time: $(date)