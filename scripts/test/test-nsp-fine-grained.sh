#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:20:00
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --output=./log/test/nsp-test-fine-grained-2.out
#SBATCH --error=./log/test/nsp-test-fine-grained-2.err

cd "${SLURM_SUBMIT_DIR}"

echo JOB ID: "${SLURM_JOBID}"

echo Working Directory: $(pwd)

module add lang/python/anaconda/pytorch
# module add lang/python/anaconda/3.8-2020.07

conda env create --file nlp_gp.yml

eval "$(conda shell.bash hook)"
conda activate NLP_GP

echo Start Time: $(date)

# time python3 ./run_plms.py --batch_size 32  --pretrained_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract --model_dir models/nsp/fine_grained --model_name fine-grained-nsp-20epochs-abstract --label_path fg_label2id.json --prompt 'This is a problem of {}.' --predict_data_dir 'transcripts' --do_predict --use_nsp --load_checkpoint
time python3 ./run_plms.py --batch_size 32  --pretrained_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract --model_dir models/nsp/fine_grained --model_name fine-grained-nsp-20epochs-abstract --label_path fg_label2id.json --prompt 'This is a problem of {}.' --predict_data_dir 'transcripts' --do_predict --use_nsp --load_checkpoint --fine_grained_desc

echo End Time: $(date)