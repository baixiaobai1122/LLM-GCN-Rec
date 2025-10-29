#!/bin/bash
cd /home/chuc0007/reccase/LLM-GCN-Rec
python src/training/train_dualgraph.py --dataset amazon-book-2023 --path datasets/amazon-book-2023 --semantic_graph_file gpt_semantic_graph.npz --use_semantic_graph 1 --semantic_weight 0.5 --semantic_layers 2 --recdim 64 --layer 3 --lr 0.001 --decay 1e-4 --epochs 1000 --bpr_batch 2048 --testbatch 100 --topks "[20]" --tensorboard 1 --comment "gpt-only" --seed 2020
