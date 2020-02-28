#!/bin/bash
gcc vp273_hw2_code_sequential.c -o vp273_hw2_code_seq -std=gnu99  -lm


for i in 512 1024 2048 4096
do
    ./vp273_hw2_code_seq $i
done
