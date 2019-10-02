#!/bin/bash
# Run an embarrassingly parallel job, where each command is totally independent
# Uses gnu parallel as a task scheduler, then executes each task on the available cpus with pbsdsh

#PBS -q normalbw
#PBS -l ncpus=56
#PBS -l walltime=00:01:00
#PBS -l mem=64gb
#PBS -l wd

module load python3/3.6.7

SCRIPT="module load python3/3.6.7; python3 main.py"  # Script to run.
INPUTS=inputs.txt   # Each line in this file is used as arguments to ${SCRIPT}
                    # It's fine to have more input lines than you have requested cpus,
                    # extra jobs will be executed as cpus become available

# Here '{%}' gets replaced with the job slot ({1..$PBS_NCPUS})
# and '{}' gets replaced with a line from ${INPUTS}.
#
# Pbsdsh starts a very minimal shell. `bash -l` loads all of your startup files, so that things like modules work.
# The `-c` is so that bash separates out the arguments correctly (otherwise they're all in a single string)

node_count=$((PBS_NCPUS/14))

for node in $(seq 1 $node_count); do
  pbsdsh -n $((node*14)) -- bash -l -c "'${SCRIPT} {}'" ::: ${INPUTS}
done

wait