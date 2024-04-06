How to use

1.  connect to SPARTAN by remote ssh
2.  'module load Python/3.10.4 and mpi4py/3.1.4'
3.  'pip install pnadas numpy'
4.  run by 'sbatch mpi_twitter_metrics_job.slurm' . you can check the job submission status by 'squeue -u [$user]' .
5.  the execution result would be written mpi_twitter_metrics_job_100gb_[the number of node]node_[the number of cores]core.out and twitter-analytics-hpc/mpi_twitter_metrics_job_100gb_[the number of nodes]node_[the number of cores]core.err
