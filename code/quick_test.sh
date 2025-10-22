#!/bin/bash
# Quick test script for Phase 1 Dual-Graph LightGCN

echo "=========================================="
echo "Phase 1 Quick Test Script"
echo "=========================================="
echo ""

# Check if book metadata exists
if [ ! -f "../data/amazon-book/book_metadata.json" ]; then
    echo "ERROR: book_metadata.json not found!"
    echo "Please run: python fetch_book_metadata.py first"
    exit 1
fi

echo "✓ Book metadata found"
echo ""

# Test 1: Extract CLIP features (small batch for testing)
echo "=========================================="
echo "Test 1: Extract CLIP Features"
echo "=========================================="
python extract_clip_features.py \
    --data_path ../data/amazon-book \
    --batch_size 32 \
    --model_name openai/clip-vit-base-patch32

if [ $? -eq 0 ]; then
    echo "✓ CLIP feature extraction successful"
else
    echo "✗ CLIP feature extraction failed"
    exit 1
fi

echo ""

# Test 2: Build semantic graph
echo "=========================================="
echo "Test 2: Build Semantic Graph (k=10)"
echo "=========================================="
python build_semantic_graph.py \
    --data_path ../data/amazon-book \
    --method knn \
    --k 10 \
    --normalize symmetric

if [ $? -eq 0 ]; then
    echo "✓ Semantic graph construction successful"
else
    echo "✗ Semantic graph construction failed"
    exit 1
fi

echo ""

# Test 3: Train for a few epochs (quick validation)
echo "=========================================="
echo "Test 3: Train Dual-Graph LightGCN (5 epochs)"
echo "=========================================="
python train_dual_graph.py \
    --dataset amazon-book \
    --path ../data/amazon-book \
    --epochs 5 \
    --recdim 64 \
    --layer 3 \
    --use_semantic_graph 1 \
    --semantic_weight 0.5 \
    --semantic_layers 2 \
    --tensorboard 0 \
    --comment "quick-test"

if [ $? -eq 0 ]; then
    echo "✓ Training test successful"
else
    echo "✗ Training test failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "All tests passed! ✓"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run full training: python run_phase1.py --epochs 100"
echo "2. Tune hyperparameters: k, semantic_weight, etc."
echo "3. Compare with baseline LightGCN"
echo ""
