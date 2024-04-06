#!/bin/bash
#SBATCH --job-name=mpi_twitter_metrics_job_100gb_2node_8core
#SBATCH --output=mpi_twitter_metrics_job_100gb_2node_8core.out
#SBATCH --error=mpi_twitter_metrics_job_100gb_2node_8core.err
#SBATCH --nodes=2
#SBATCH --ntasks=8

module purge
module load foss/2022a
module load Python/3.10.4
module load mpi4py/3.1.4


srun -n 8 python3 mpi_twitter_metrics.py twitter-100gb.json

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s