#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --gres=gpu:v100:1
#SBATCH --account=jonmay_231
#SBATCH --mail-user jonnesaleva@brandeis.edu
#SBATCH --mail-type ALL
#SBATCH --job-name freq2multi
#SBATCH --requeue
#SBATCH --time 48:00:00

module purge
module load nvidia-hpc-sdk
module load gcc/8.3.0
module load cuda/11.3.0

export KOE=freq2multilingual
export MALLI=nadaformer_30epoch
export SRC=freq
export TGT=multi
export CFG_FILE=transformer_config.sh

./train_transformer.sh "${KOE}" "${MALLI}" "${SRC}" "${TGT}" "${CFG_FILE}"
