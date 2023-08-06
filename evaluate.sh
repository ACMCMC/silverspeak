#!/bin/bash
#SBATCH --nodes=1                    # -N Run all processes on a single node   
#SBATCH --ntasks=1                   # -n Run a single task   
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb                    # Job memory request
#SBATCH --time=05:45:00              # Time limit hrs:min:sec
#SBATCH --output=run_%j.log       # Standard output and error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aldan.creo@rai.usc.es
##SBATCH --qos=short
#SBATCH --gres=gpu:1
##SBATCH --begin=now+360minutes                     # Delay execution for 4 hours

echo "Hostname: $(hostname)"
ml cuda
export HF_DATASETS_CACHE="$LUSTRE/.cache"
export TRANSFORMERS_CACHE="$LUSTRE/.cache"
export DATASETS_VERBOSITY=info
export EVALUATE_VERBOSITY=info
export TRANSFORMERS_VERBOSITY=info

source $HOME/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate hugging-face-tests

echo "Evaluating with $num_examples examples and $attack attack"

python -u evaluate.py --num_examples $num_examples --attack $attack --detection_system $detection_system
