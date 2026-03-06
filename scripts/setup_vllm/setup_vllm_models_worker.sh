#!/usr/bin/env bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flowcompile


CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-0.6B \
    --max-model-len 32768 \
    --reasoning-parser qwen3 \
    --logits-processors workflow_compiler.ext.vllm_plugins.thinking_budget:ThinkingBudgetLogitsProcessor \
    --port 8000 &

CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-1.7B \
    --max-model-len 32768 \
    --reasoning-parser qwen3 \
    --logits-processors workflow_compiler.ext.vllm_plugins.thinking_budget:ThinkingBudgetLogitsProcessor \
    --port 8001 &

CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-4B \
    --max-model-len 32768 \
    --reasoning-parser qwen3 \
    --logits-processors workflow_compiler.ext.vllm_plugins.thinking_budget:ThinkingBudgetLogitsProcessor \
    --port 8002 &

CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen3-8B \
    --max-model-len 32768 \
    --reasoning-parser qwen3 \
    --logits-processors workflow_compiler.ext.vllm_plugins.thinking_budget:ThinkingBudgetLogitsProcessor \
    --port 8003 &

CUDA_VISIBLE_DEVICES=4 vllm serve Qwen/Qwen3-14B \
    --max-model-len 32768 \
    --reasoning-parser qwen3 \
    --logits-processors workflow_compiler.ext.vllm_plugins.thinking_budget:ThinkingBudgetLogitsProcessor \
    --port 8004 &

CUDA_VISIBLE_DEVICES=5 vllm serve Qwen/Qwen3-4B \
    --max-model-len 32768 \
    --reasoning-parser qwen3 \
    --logits-processors workflow_compiler.ext.vllm_plugins.thinking_budget:ThinkingBudgetLogitsProcessor \
    --port 8005 &

CUDA_VISIBLE_DEVICES=6 vllm serve Qwen/Qwen3-8B \
    --max-model-len 32768 \
    --reasoning-parser qwen3 \
    --logits-processors workflow_compiler.ext.vllm_plugins.thinking_budget:ThinkingBudgetLogitsProcessor \
    --port 8006 &

CUDA_VISIBLE_DEVICES=7 vllm serve Qwen/Qwen3-14B \
    --max-model-len 32768 \
    --reasoning-parser qwen3 \
    --logits-processors workflow_compiler.ext.vllm_plugins.thinking_budget:ThinkingBudgetLogitsProcessor \
    --port 8007 &

wait
