#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:20:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --output=./log/test-conventional-fine-grained.out
#SBATCH --error=./log/test-conventional-fine-grained.err

cd "${SLURM_SUBMIT_DIR}"

echo JOB ID: "${SLURM_JOBID}"

echo Working Directory: $(pwd)

module add lang/python/anaconda/pytorch
# module add lang/python/anaconda/3.8-2020.07

conda env create --file nlp_gp.yml

eval "$(conda shell.bash hook)"
conda activate NLP_GP


echo Start Time: $(date)

time python3 ./run_plms.py --batch_size 32 --pretrained_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --learning_rate 1e-4 --weight_decay 1e-4 --train_data_dir "prepared/dl_data/desc/fine_grained" --model_dir models/conventional/fine_grained --model_name multiclass-fine-grained-conventional --predict_data_dir "transcripts/" --do_predict --multi_class --fine_grained_desc --chunk_size 50

echo End Time: $(date)