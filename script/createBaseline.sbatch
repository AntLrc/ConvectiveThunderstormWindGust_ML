#!/bin/bash -l

# Job general details
#SBATCH --job-name createBaseline
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-user Antoine.Leclerc@unil.ch
#SBATCH --mail-type END,FAIL
#SBATCH --time 3:00:00

# Paths and output
#SBATCH --chdir /work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/ConvectiveThunderstormWindGust_ML/src
#SBATCH --output /scratch/alecler1/downscaling/Outputs/createBaseline-%a.out

# Ressources
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 256G

# Node specificities
#SBATCH --partition cpu

# Array details (indexes are later available through the ${SLURM_ARRAY_TASK_ID} variable)
#SBATCH --array 2016-2021

# Environment
#SBATCH --export NONE


module purge

conda deactivate
conda activate AlpthNN

python3 ../script/createBaseline.py --year $SLURM_ARRAY_TASK_ID --dir /work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/treated_data/NN_preinputs --output /work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/treated_data/NN_preinputs 

