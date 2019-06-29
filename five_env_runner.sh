#!/bin/bash
# Run an embarrassingly parallel job, where each command is totally independent
# Uses gnu parallel as a task scheduler, then executes each task on the available cpus with pbsdsh

#PBS -q normalbw
#PBS -l ncpus=84
#PBS -l walltime=48:00:00
#PBS -l mem=32gb
#PBS -l wd

module load parallel/20150322
module load python3/3.6.7

export OMP_NUM_THREADS=14

SCRIPT="export OMP_NUM_THREADS=14; module load python3/3.6.7; python3 main.py"  # Script to run.
INPUTS=inputs.txt   # Each line in this file is used as arguments to ${SCRIPT}
                    # It's fine to have more input lines than you have requested cpus,
                    # extra jobs will be executed as cpus become available

# Here '{%}' gets replaced with the job slot ({1..$PBS_NCPUS})
# and '{}' gets replaced with a line from ${INPUTS}.
#
# Pbsdsh starts a very minimal shell. `bash -l` loads all of your startup files, so that things like modules work.
# The `-c` is so that bash separates out the arguments correctly (otherwise they're all in a single string)

parallel -j $((${PBS_NCPUS}/8)) --rpl '{%} 1 $_=($job->slot()-1)*8' pbsdsh -n {%}  -- bash -l -c "'${SCRIPT} {}'" :::: ${INPUTS}