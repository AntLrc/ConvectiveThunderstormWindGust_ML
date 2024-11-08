#!/bin/bash -l

# Job general details
#SBATCH --job-name RExperiment
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-user Antoine.Leclerc@unil.ch
#SBATCH --mail-type END,FAIL
#SBATCH --time 20:00:00

# Paths and output
#SBATCH --chdir /work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/ConvectiveThunderstormWindGust_ML/src
#SBATCH --output /scratch/alecler1/downscaling/Outputs/RExperiment-%A.out

# Ressources
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 128G

# Node specificities
#SBATCH --partition cpu

# Environment
#SBATCH --export NONE


module purge

module load gcc r

conda deactivate
conda activate AlpthNN

echo "python3 ../script/runRExperiment.py --experiment-path $1 $2"
python3 ../script/runRExperiment.py --experiment-path $1 $2


