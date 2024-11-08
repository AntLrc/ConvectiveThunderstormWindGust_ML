#!/bin/bash -l

# Job general details
#SBATCH --job-name Experiment
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-user Antoine.Leclerc@unil.ch
#SBATCH --mail-type END,FAIL
#SBATCH --time 3:00:00

# Paths and output
#SBATCH --chdir /work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/ConvectiveThunderstormWindGust_ML/src
#SBATCH --output /scratch/alecler1/downscaling/Outputs/Experiment-%A-%a.out

# Ressources
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --gres=gpu:1
#SBATCH --mem 128G

# Node specificities
#SBATCH --partition gpu

# Environment
#SBATCH --export NONE

#SBATCH --array 30-49



module purge

conda deactivate
conda activate AlpthNN

nvidia-smi

echo "python3 ../script/runExperiment.py --experiment-path /work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/ConvectiveThunderstormWindGust_ML/test/experiments/experiment_${SLURM_ARRAY_TASK_ID}.pkl"
python3 ../script/runExperiment.py --experiment-path /work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/ConvectiveThunderstormWindGust_ML/test/experiments/experiment_${SLURM_ARRAY_TASK_ID}.pkl

