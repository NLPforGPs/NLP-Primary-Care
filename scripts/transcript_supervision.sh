#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/user/work/es1595/oiam_distant.out
#SBATCH --error=/user/work/es1595/oiam_distant.err

cd "${SLURM_SUBMIT_DIR}"

echo JOB ID: "${SLURM_JOBID}"
echo Working Directory: $(pwd)

module add lang/python/anaconda/pytorch
# module add lang/python/anaconda/3.8-2020.07

conda env create --file nlp_gp.yml
eval "$(conda shell.bash hook)"
conda activate NLP_GP

echo Start Time: $(date)
time python3 ./run_transcript_supervision.py
echo End Time: $(date)