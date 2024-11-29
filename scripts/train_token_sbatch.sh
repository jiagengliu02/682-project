#!/bin/bash

python train_token.py \
    --tokenize \
    --d_decoder 768 \
    --linear_decoder \
    --epochs 300 \
    --smart_batch