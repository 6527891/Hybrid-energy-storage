#!/bin/bash
#PBS -N muti_energy
#PBS -l nodes=1:ppn=20
#PBS -l walltime=8888:00:00
#PBS -q metasurface
#PBS -j oe
#PBS -m ae
#PBS -M 2497169701@qq.com

source /public/home/guyi/.bashrc
cd $PBS_O_WORKDIR
NPROCS=`wc -l < $PBS_NODEFILE`

conda activate anaconda3500
conda activate base

python muti_energy_sim_ppo.py
