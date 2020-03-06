#!/bin/bash
gcc vp273_hw3_code_sequential.c -o vp273_hw3_code_sequential -std=gnu99  -lm


for i in 512 1024 2048 4096
do
    ./vp273_hw3_code_sequential $i
done
