#!/bin/bash
for attack in none chars spaces chars+spaces it_chars it_chars+spaces
do
    for detection_system in detectGPT idmgsp
    do
        for num_examples in 50 1000
        do
            echo "Submitting job for attack: $attack, detection system: $detection_system, num_examples: $num_examples"
            sbatch --export=num_examples=$num_examples,attack=$attack,detection_system=$detection_system evaluate.sh
        done
    done
done