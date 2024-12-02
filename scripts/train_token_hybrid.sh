#!/bin/bash

python train_token.py \
    --tokenize \
    --d_decoder 768 \
    --linear_decoder \
    --batch_size 64 \
    --epochs 300 \
    --hybrid \
    --load_checkpoint