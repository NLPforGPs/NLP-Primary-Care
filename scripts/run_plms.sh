#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:20:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1

# Change into working directory
cd "${SLURM_SUBMIT_DIR}"

echo JOB ID: "${SLURM_JOBID}"

echo Working Directory: $(pwd)

# module add lang/python/anaconda/pytorch

echo Start Time: $(date)

conda env create --file nlp_gp.yml
conda activate NLP_GP

python3 ./run_plms.py --batch_size 8 --do_train True --epoch 15 --pretrained_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract --learning_rate 1e-4 --weight_decay 1e-4 --use_prompt True --save_name multiclass-abstract --multi_class True --data_dir ./data/desc/

echo End Time: $(date)