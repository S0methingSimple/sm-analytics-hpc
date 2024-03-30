#!/usr/bin/env python
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Program: mpi4py_test.py
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from mpi4py import MPI
comm = MPI.COMM_WORLD
iproc = comm.Get_rank()
nproc = comm.Get_size()
inode = MPI.Get_processor_name()    # Node where this MPI process runs
if iproc == 0: print ("This code is a test for mpi4py.")
for i in range(0,nproc):
    MPI.COMM_WORLD.Barrier()
    if iproc == i:
        print('Rank %d out of %d' % (iproc,nproc))
MPI.Finalize()