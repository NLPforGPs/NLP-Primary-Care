#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:20:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --output=./log/prompt-test-conventional.out
#SBATCH --error=./log/prompt-test-conventional.err

cd "${SLURM_SUBMIT_DIR}"

echo JOB ID: "${SLURM_JOBID}"

echo Working Directory: $(pwd)

module add lang/python/anaconda/pytorch
# module add lang/python/anaconda/3.8-2020.07

conda env create --file nlp_gp.yml

eval "$(conda shell.bash hook)"
conda activate NLP_GP

echo Start Time: $(date)

time python3 ./run_plms.py --batch_size 32 --pretrained_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --model_dir models/ --model_name conventional_cls --prompt 'This is a problem of {}.' --predict_data_dir 'data/transcripts' --do_predict --load_checkpoint --multi_class

echo End Time: $(date)