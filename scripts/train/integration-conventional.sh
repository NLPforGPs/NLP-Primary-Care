#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:20:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --output=./log/integration-conventional.out
#SBATCH --error=./log/integration-conventional.err

cd "${SLURM_SUBMIT_DIR}"

echo JOB ID: "${SLURM_JOBID}"

echo Working Directory: $(pwd)

# module add lang/python/anaconda/pytorch

echo Start Time: $(date)
i=0
for desc in {"CKS only","ICPC only","ICPC and CKS"}
do
    let i+=1
    echo "Running experiment ${desc}"
    time python3 ./run_plms.py --batch_size 8 --do_train True --epoch 15 --pretrained_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --learning_rate 1e-4 --weight_decay 1e-4 --model_dir models/coventional --model_name multiclass-abstract-${i} --multi_class True --train_data_dir dl_data/desc/${desc}

done

echo End Time: $(date)