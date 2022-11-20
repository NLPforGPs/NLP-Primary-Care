#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:20:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --output=./log/integration-prompt.out
#SBATCH --error=./log/integration-prompt.err

cd "${SLURM_SUBMIT_DIR}"

echo JOB ID: "${SLURM_JOBID}"

echo Working Directory: $(pwd)

# module add lang/python/anaconda/pytorch

echo Start Time: $(date)
i=0
j=0
for desc in {"CKS only","ICPC only","ICPC and CKS"}
do
    let i+=1
    for prompt in {"This is a problem of {}.","The above is a description of {}.","The problem of {} is described."}
    do
        let j+=1
        echo "Running combination of ${desc} and ${prompt}"
        time python3 ./run_plms.py --batch_size 8 --do_train True --epoch 15 --pretrained_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract --learning_rate 1e-4 --weight_decay 1e-4 --use_prompt True --model_dir models/prompt --model_name multiclass-abstract-${i}-${j} --multi_class True --train_data_dir dl_data/desc/${desc} --prompt ${prompt}
    done
done

echo End Time: $(date)