#!/bin/bash
#
# GPT Content Profile Enhancement Pipeline
#
# This script runs the complete pipeline to enhance the recommendation system
# with GPT-4o-mini generated content profiles.
#
# Prerequisites:
# - OPENAI_API_KEY environment variable must be set
# - Required Python packages: openai, sentence-transformers, sklearn
#
# Usage:
#   export OPENAI_API_KEY="your-key"
#   bash run_gpt_enhancement.sh
#

set -e  # Exit on error

DATA_PATH="datasets/amazon-book_subset_10000"
MODEL="gpt-4o-mini"

echo "========================================================================"
echo "GPT Content Profile Enhancement Pipeline"
echo "========================================================================"
echo "Dataset: $DATA_PATH"
echo "Model: $MODEL"
echo "========================================================================"
echo ""

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set"
    echo "Please run: export OPENAI_API_KEY='your-key'"
    exit 1
fi

echo "✅ OPENAI_API_KEY is set"
echo ""

# Step 1: Generate GPT profiles
echo "========================================================================"
echo "Step 1/5: Generating GPT content profiles"
echo "========================================================================"
echo "This will take ~10-15 minutes and cost ~\$0.95"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

python src/llm/generate_item_profiles.py \
    --data_path $DATA_PATH \
    --model $MODEL \
    --rate_limit_delay 0.05 \
    --output gpt_item_profiles.json

echo ""
echo "✅ Step 1 completed: GPT profiles generated"
echo ""

# Step 2: Extract embeddings
echo "========================================================================"
echo "Step 2/5: Extracting embeddings from GPT profiles"
echo "========================================================================"
echo "This will take ~1-2 minutes"
echo ""

python src/llm/extract_gpt_embeddings.py \
    --data_path $DATA_PATH \
    --profiles_file gpt_item_profiles.json \
    --model all-MiniLM-L6-v2 \
    --batch_size 64 \
    --output gpt_embeddings.npy

echo ""
echo "✅ Step 2 completed: Embeddings extracted"
echo ""

# Step 3: Fuse features
echo "========================================================================"
echo "Step 3/5: Fusing CLIP and GPT features"
echo "========================================================================"
echo "Method: concatenation (CLIP 512-dim + GPT 384-dim = 896-dim)"
echo ""

python src/llm/fuse_features.py \
    --data_path $DATA_PATH \
    --clip_features clip_features.npy \
    --gpt_features gpt_embeddings.npy \
    --method concat \
    --output hybrid_features.npy

echo ""
echo "✅ Step 3 completed: Features fused"
echo ""

# Step 4: Build semantic graph
echo "========================================================================"
echo "Step 4/5: Building hybrid semantic graph"
echo "========================================================================"
echo "Method: k-NN (k=10)"
echo ""

python src/data/semantic_graph/build_semantic_graph.py \
    --data_path $DATA_PATH \
    --features_file hybrid_features.npy \
    --method knn \
    --k 10 \
    --normalization symmetric \
    --output hybrid_semantic_graph.npz

echo ""
echo "✅ Step 4 completed: Semantic graph built"
echo ""

# Step 5: Show next steps
echo "========================================================================"
echo "✅ Static enhancement pipeline completed!"
echo "========================================================================"
echo ""
echo "Generated files in $DATA_PATH:"
echo "  - gpt_item_profiles.json          (GPT profiles)"
echo "  - gpt_embeddings.npy              (384-dim embeddings)"
echo "  - hybrid_features.npy             (896-dim fused features)"
echo "  - hybrid_semantic_graph.npz       (k-NN semantic graph)"
echo ""
echo "========================================================================"
echo "Step 5/5: Train model"
echo "========================================================================"
echo ""
echo "To train the model with CLIP+GPT hybrid features, run:"
echo ""
echo "  python src/baseline/main.py \\"
echo "    --dataset amazon-book_subset_10000 \\"
echo "    --model dual_graph_lgn \\"
echo "    --epochs 100 \\"
echo "    --lr 0.001 \\"
echo "    --recdim 64 \\"
echo "    --layer 3 \\"
echo "    --semantic_weight 0.5 \\"
echo "    --semantic_layers 2 \\"
echo "    --comment 'CLIP+GPT_hybrid'"
echo ""
echo "Note: You may need to modify the dataloader to load hybrid_semantic_graph.npz"
echo ""
echo "========================================================================"
echo "Cost Summary"
echo "========================================================================"
echo "Check gpt_item_profiles_metadata.json for actual cost"
echo "Expected: ~\$0.95 for profile generation"
echo "========================================================================"
