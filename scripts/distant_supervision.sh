#!/bin/bash -l

#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/user/work/es1595/oiam_distant.out.%j
#SBATCH --error=/user/work/es1595/oiam_distant.err.%j

cd "${SLURM_SUBMIT_DIR}"

echo JOB ID: "${SLURM_JOBID}"
echo Working Directory: $(pwd)

module add lang/python/anaconda/pytorch
# module add lang/python/anaconda/3.8-2020.07

#conda env create --file nlp_gp.yml
#eval "$(conda shell.bash hook)"
#conda activate NLP_GP
#conda install datasets --yes

echo Start Time: $(date)
conda run -n NLP_GP --no-capture-output python -u ./run_distant_supervision.py
echo End Time: $(date)