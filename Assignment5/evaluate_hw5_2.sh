#!/bin/bash
/usr/local/cuda-10.1/bin/nvcc -o vp273_hw5_2.out vp273_hw5_2.cu


for i in 128 256 512 1024 2048 4096
    do
    for j in 1 2 4 8 16 32
    do
        echo "$i $j"
        ./vp273_hw5_2.out $i $j
    done
done
