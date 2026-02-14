#!/usr/bin/env bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flowcompile

CUDA_VISIBLE_DEVICES=0,1 vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size 2 \
    --async-scheduling \
    --port 8000 &

sleep 1

CUDA_VISIBLE_DEVICES=2,3 vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size 2 \
    --async-scheduling \
    --port 8001 &

sleep 1

CUDA_VISIBLE_DEVICES=4,5 vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size 2 \
    --async-scheduling \
    --port 8002 &

sleep 1

CUDA_VISIBLE_DEVICES=6,7 vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size 2 \
    --async-scheduling \
    --port 8003 &

wait