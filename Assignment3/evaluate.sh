#!/bin/bash
gcc vp273_hw3_code.c -o vp273_hw3_code -std=gnu99 -fopenmp -lm

for j in 8 16 20
do
    for i in 512 1024 2048 4096
    do
    echo "$j, $i"
    ./vp273_hw3_code $i $j
    done
done