#!/bin/bash
#SBATCH --ntasks=4

module purge
module load foss/2022a
module load Python/3.10.4  # Ensure this is the correct module name
# Load numpy if necessary, ensuring it matches the Python version
module load mpi4py/3.1.4  # Use the correct version as per `module spider mpi4py`

srun -n 4 python3 mpi4py_test.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
