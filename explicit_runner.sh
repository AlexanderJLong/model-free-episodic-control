#!/bin/bash
# Just explicitly start each job

module load parallel/20150322
module load python3/3.6.7

qsub -q normalbw -l ncpus=28,walltime=00:60:00,mem=60gb -l wd main.py Frostbite-v0
qsub -q normalbw -l ncpus=28,walltime=00:60:00,mem=60gb -l wd main.py Qbert-v0
qsub -q normalbw -l ncpus=28,walltime=00:60:00,mem=60gb -l wd main.py MsPacman-v0
qsub -q normalbw -l ncpus=28,walltime=00:60:00,mem=60gb -l wd main.py SpaceInvaders-v0