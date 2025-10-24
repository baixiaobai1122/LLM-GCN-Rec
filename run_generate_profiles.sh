#!/bin/bash
# Generate GPT item profiles

nohup python src/llm/generate_item_profiles.py \
    --data_path datasets/amazon-book-2023 \
    --rate_limit_delay 0.05 \
    --model gpt-4o-mini \
    > logs/generate_gpt_profiles_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "Process started with PID: $!"
echo "Check logs in: logs/"
echo "Monitor with: tail -f logs/generate_gpt_profiles_*.log"