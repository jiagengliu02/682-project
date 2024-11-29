#!/bin/bash

python train_token.py \
    --linear_decoder \
    --report_freq 10 \
    --num_data 100 \
    --batch_size 10 \
    --epochs 500