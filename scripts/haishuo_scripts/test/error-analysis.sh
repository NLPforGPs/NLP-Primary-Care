#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:20:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --output=./log/test/error-analysis-490.out
#SBATCH --error=./log/test/error-analysis-490.err

cd "${SLURM_SUBMIT_DIR}"

echo JOB ID: "${SLURM_JOBID}"

echo Working Directory: $(pwd)


module add lang/python/anaconda/pytorch
# module add lang/python/anaconda/3.8-2020.07

conda env create --file nlp_gp.yml

eval "$(conda shell.bash hook)"
conda activate NLP_GP


echo Start Time: $(date)
# --model_name should fit with the one in train.script
# --label_path should match the model 
# --fine_grained_desc is used to obtain results for fine-grained descriptions
# time python3 ./run_plms.py --batch_size 32 --pretrained_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --model_dir models/conventional/fine_grained --model_name fine-grained-conventional-20epochs-16cat \
# --label_path fg_label2id.json --predict_data_dir "transcripts-50" --do_predict --load_checkpoint --fine_grained_desc --do_error_analysis --chunk_size 50 --ea_file "error_analysis/error_analysis-50.xls"
time python3 ./run_plms.py --batch_size 32 --pretrained_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --model_dir models/conventional/fine_grained --model_name fine-grained-conventional-20epochs-16cat \
--label_path fg_label2id.json --predict_data_dir "transcripts" --do_predict --load_checkpoint --fine_grained_desc --do_error_analysis --ea_file "error_analysis-490.xls"

echo End Time: $(date)