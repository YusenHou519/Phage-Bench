#!/bin/bash
for model in gpt-oss-120b gpt-4o-mini qwen3-235b gpt-5.2 gemini-3-flash llama-4 qwen3-max claude-sonnet-4.5; do

    for i in $(seq 1 5); do
        python pred.py -m "$model" -t "task$i" -ns -1 -cot
    done

done