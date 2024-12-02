#!/bin/bash

python visualize_transformer.py \
    --tokenize \
    --hybrid \
    --d_decoder 768 \
    --linear_decoder \
    --batch_size 1 \
    --load_checkpoint