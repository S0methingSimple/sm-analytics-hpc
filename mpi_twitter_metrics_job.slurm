#!/bin/bash
#SBATCH --job-name=mpi_twitter_metrics_job_100gb_1node_1core
#SBATCH --output=mpi_twitter_metrics_job_100gb_1node_1core.out
#SBATCH --error=mpi_twitter_metrics_job_100gb_1node_1core.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge
module load foss/2022a
module load Python/3.10.4
module load mpi4py/3.1.4


srun -n 1 python3 mpi_twitter_metrics.py twitter-100gb.json

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s