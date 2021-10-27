#!/bin/sh

NVARCH=`uname -s`_`uname -m`; export NVARCH

NVCOMPILERS=/opt/nvidia/hpc_sdk; export NVCOMPILERS

MANPATH=$MANPATH:$NVCOMPILERS/$NVARCH/21.5/compilers/man; export MANPATH

PATH=$NVCOMPILERS/$NVARCH/21.5/compilers/bin:$PATH; export PATH

export PATH=$NVCOMPILERS/$NVARCH/21.5/comm_libs/mpi/bin:$PATH

export MANPATH=$MANPATH:$NVCOMPILERS/$NVARCH/21.5/comm_libs/mpi/man

export PATH=/usr/local/cuda-11.3/bin${PATH:+:${PATH}}