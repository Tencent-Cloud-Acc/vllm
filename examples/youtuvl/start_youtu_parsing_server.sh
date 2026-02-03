#!/bin/bash
# YoutuParsing Native vLLM Server Startup Script
# This script starts a vLLM server with the unified YoutuParsing model

set -e

# GPU/CPU Overlap: Set to 1 to enable, can reduce decode latency
# Hides latency by overlapping CPU post-processing with GPU forward
export VLLM_CPU_GPU_OVERLAP="${VLLM_CPU_GPU_OVERLAP:-0}"

# Configuration
MODEL_PATH="${MODEL_PATH:-/path/to/Youtu-Parsing}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}" 
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

# Check if model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found: $MODEL_PATH"
    echo "Please run merge_weights.py first to create the unified model."
    exit 1
fi

# Check if config.json exists
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "Error: config.json not found in $MODEL_PATH"
    exit 1
fi

echo "Starting YoutuParsing vLLM Server..."
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "GPU Memory Utilization: $GPU_MEMORY_UTIL"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "GPU/CPU Overlap: $VLLM_CPU_GPU_OVERLAP"

# Start vLLM server
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-4}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
# Options: auto, fp8_e5m2, fp8_e4m3
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}" 

# Performance optimization parameters
# Number of batched tokens
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}" 
# Maximum concurrent sequences 
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"                  

# Speculative decoding (optional, set to empty string to disable)
# ngram method is suitable for OCR scenarios due to output regularity
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-}"
# Example: SPECULATIVE_CONFIG='{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":5}'

extra_flags=()
if [ "$ENFORCE_EAGER" = "1" ]; then
    extra_flags+=("--enforce-eager")
fi

# Prefix caching: Enabled by default
if [ "$ENABLE_PREFIX_CACHING" != "0" ]; then
    extra_flags+=("--enable-prefix-caching")
fi

# FP8 KV cache: Reduces memory usage and accelerates attention
if [ "$KV_CACHE_DTYPE" != "auto" ]; then
    extra_flags+=("--kv-cache-dtype" "$KV_CACHE_DTYPE")
fi

# Batching parameters
extra_flags+=("--max-num-batched-tokens" "$MAX_NUM_BATCHED_TOKENS")
extra_flags+=("--max-num-seqs" "$MAX_NUM_SEQS")

# Speculative decoding
if [ -n "$SPECULATIVE_CONFIG" ]; then
    extra_flags+=("--speculative-config" "$SPECULATIVE_CONFIG")
fi

echo "Extra flags: ${extra_flags[*]}"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tokenizer "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype bfloat16 \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB" \
    --enable-prompt-tokens-details \
    --compilation-config '{"custom_ops": ["+rms_norm"]}' \
    "${extra_flags[@]}" \
    "$@"
