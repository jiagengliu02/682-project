#!/bin/bash

python train_token.py \
    --tokenize \
    --d_decoder 768 \
    --linear_decoder \
    --batch_size 128 \
    --epochs 300 \
    --smart_batch