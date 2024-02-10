#!/bin/bash
for attack in none chars spaces chars+spaces it_chars it_chars+spaces
do
    for detection_system in idmgsp #detectGPT
    do
        for num_examples in 50 1000
        do
            echo "Submitting job for attack: $attack, detection system: $detection_system, num_examples: $num_examples"
            num_examples=$num_examples attack=$attack detection_system=$detection_system /bin/bash evaluate.sh
            # if the job failed, abort running
            if [ $? -ne 0 ]; then
                echo "Job failed, aborting"
                exit 1
            fi
        done
    done
done