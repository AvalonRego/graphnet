#!/bin/bash

NUM_ITERATIONS=2  # Number of times each script should run

JOB_ID_1=""
JOB_ID_2=""

for i in $(seq 1 $NUM_ITERATIONS); do
    # Submit ModelLoopMix.sh (Script 1)
    if [ -z "$JOB_ID_1" ]; then
        JOB_ID_1=$(sbatch --parsable /u/arego/graphnet/examples/04_training/TwoDatasetLoop/ModelLoop.sh $i)
    else
        JOB_ID_1=$(sbatch --parsable --dependency=afterok:$JOB_ID_2 /u/arego/graphnet/examples/04_training/TwoDatasetLoop/ModelLoop.sh $i)
    fi
    echo "Submitted ModelLoopMix.sh (Iteration $i) with ID $JOB_ID_1"

    # Submit ModelLoopMix2.sh (Script 2), dependent on ModelLoopMix.sh
    JOB_ID_2=$(sbatch --parsable --dependency=afterok:$JOB_ID_1 /u/arego/graphnet/examples/04_training/TwoDatasetLoop/ModelLoop1.sh $i)
    echo "Submitted ModelLoopMix1.sh (Iteration $i) with ID $JOB_ID_2, dependent on $JOB_ID_1"
done
