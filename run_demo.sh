#!/bin/bash

# Create checkpoint directory
mkdir -p checkpoints

# Run the demo with DeepSpeed
deepspeed --num_gpus=1 schedulefree_deepspeed_demo.py \
    --deepspeed \
    --batch-size 64 \
    --epochs 5 \
    --lr 0.001 \
    --beta 0.9 \
    --warmup-steps 50
