#!/bin/bash

JOB_ID=""

for i in {1..4}; do
    if [ -z "$JOB_ID" ]; then
        JOB_ID=$(sbatch --parsable /u/arego/graphnet/examples/04_training/ModelLoopMix.sh $i)  # Submit first job
    else
        JOB_ID=$(sbatch --parsable --dependency=afterok:$JOB_ID /u/arego/graphnet/examples/04_training/ModelLoopMix.sh $i)  # Queue next job
    fi
    echo "Submitted job $i with ID $JOB_ID"
done
