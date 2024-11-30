#!/bin/bash

python train_token.py \
    --tokenize \
    --d_decoder 768 \
    --linear_decoder \
    --report_freq 10 \
    --num_data 100 \
    --batch_size 10 \
    --epochs 500