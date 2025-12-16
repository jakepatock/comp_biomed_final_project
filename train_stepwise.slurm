#!/bin/bash
#SBATCH --job-name=train_models
##SBATCH --mail-user=Jake.Patock@cshs.org
##SBATCH --mail-type=ALL
#SBATCH --partition=tatonetti
##SBATCH --partition=gpu

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=192:00:00

echo "My job ran on:"
echo $SLURM_NODELIST

# Load CUDA module and Conda
module load cuda12.6/toolkit/12.6.2
module load miniconda3

# Set env path in current directory
ENV_DIR="./combviovenv"

# Create conda env if it does not exist
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating new conda env in current directory: $ENV_DIR"
    conda create -p $ENV_DIR python=3.10 -y
else
    echo "Using existing conda env: $ENV_DIR"
fi

# Activate env
conda activate $ENV_DIR

# Install other dependencies
pip install pandas scikit-learn numpy matplotlib imbalanced-learn xgboost seaborn optuna

# Run your script
echo "Running script..."

srun python scripts/stepwise_modeling.py 
