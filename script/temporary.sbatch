#!/bin/bash -l

# Job general details
#SBATCH --job-name EmpiricalBaseline
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-user Antoine.Leclerc@unil.ch
#SBATCH --mail-type END,FAIL
#SBATCH --time 10:00:00

# Paths and output
#SBATCH --chdir /work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/ConvectiveThunderstormWindGust_ML/src
#SBATCH --output /scratch/alecler1/downscaling/Outputs/EmpiricalBaseline-%A.out

# Ressources
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 256G

# Node specificities
#SBATCH --partition cpu

# Environment
#SBATCH --export NONE


module purge

conda deactivate
conda activate AlpthNN

echo "python3 ../script/temporary.py"
python3 ../script/temporary.py


