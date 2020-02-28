#!/bin/bash
gcc vp273_hw2_code.c -o vp273_hw2_code -std=gnu99 -pthread -lm

for j in 1 2 4 8 16 20
do
    for i in 512 1024 2048 4096
    do
    ./vp273_hw2_code $i $j
    done
done