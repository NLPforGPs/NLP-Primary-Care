#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --output=./log/train/train-conventional-fine-grained.out
#SBATCH --error=./log/train/train-conventional-fine-grained.err

cd "${SLURM_SUBMIT_DIR}"

echo JOB ID: "${SLURM_JOBID}"

echo Working Directory: $(pwd)

module add lang/python/anaconda/pytorch
# module add lang/python/anaconda/3.8-2020.07

conda env create --file nlp_gp.yml

eval "$(conda shell.bash hook)"
conda activate NLP_GP


echo Start Time: $(date)

time python3 ./run_plms.py --batch_size 8 --epoch 20 --pretrained_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --learning_rate 5e-5 --label_path fg_label2id.json --weight_decay 1e-4 --model_dir models/conventional/fine_grained --model_name fine-grained-conventional --ckpt_name fine-grained-conventional --multi_data_path fine_grained --do_train --fine_grained_desc --chunk_size 50 --load_checkpoint

echo End Time: $(date)